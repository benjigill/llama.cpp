// build or extend an ngram-mod table file (--spec-ngram-mod-file) from a text corpus
//
// the corpus is tokenized with the vocab of the target model (special tokens are parsed, as in server
// prompts), so chat transcripts rendered with the chat template match what the server sees.
// documents are separated by NUL bytes; n-grams never span two documents. later documents overwrite
// earlier ones in a shared bucket, so put the most relevant material last.

#include "common.h"
#include "ngram-mod.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <model.gguf> -o <table.bin> [options] <corpus>...\n"
        "\n"
        "  -m FNAME        model whose vocab is used (only the vocab is loaded)\n"
        "  -o FNAME        table file to write\n"
        "  --size MiB      table size for a new table, 8 bytes per n-gram (default: 16)\n"
        "  --n-match N     n-gram length, must match --spec-ngram-mod-n-match of the server (default: 24)\n"
        "  --append        extend the table in -o if it exists (its size is kept)\n"
        "  <corpus>        text files, '-' for stdin; documents are separated by NUL bytes\n",
        argv0);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string fname_model;
    std::string fname_out;
    int32_t     size_mib = 16;
    int32_t     n_match  = 24;
    bool        append   = false;

    std::vector<std::string> inputs;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_val = i + 1 < argc;

        if (arg == "-m" && has_val) {
            fname_model = argv[++i];
        } else if (arg == "-o" && has_val) {
            fname_out = argv[++i];
        } else if (arg == "--size" && has_val) {
            size_mib = std::atoi(argv[++i]);
        } else if (arg == "--n-match" && has_val) {
            n_match = std::atoi(argv[++i]);
        } else if (arg == "--append") {
            append = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-" || arg[0] != '-') {
            inputs.push_back(arg);
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (fname_model.empty() || fname_out.empty() || inputs.empty() || size_mib < 1 || size_mib > 16384 || n_match < 1) {
        print_usage(argv[0]);
        return 1;
    }

    common_init();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(fname_model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load the vocab from %s\n", fname_model.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_vocab = llama_vocab_n_tokens(vocab);

    common_ngram_mod mod(n_match, (size_t) size_mib*1024*1024/sizeof(common_ngram_mod::cell_t));

    std::string err;
    if (append) {
        if (FILE * f = std::fopen(fname_out.c_str(), "rb")) {
            std::fclose(f);
            if (!mod.load(fname_out, n_vocab, err)) {
                fprintf(stderr, "cannot extend %s: %s\n", fname_out.c_str(), err.c_str());
                return 1;
            }
            fprintf(stderr, "extending %s: %zu/%zu cells used\n", fname_out.c_str(), mod.get_used(), mod.size());
        }
    }

    const size_t n = mod.get_n();

    size_t n_docs   = 0;
    size_t n_tokens = 0;
    size_t n_bytes  = 0;

    auto add_doc = [&](const std::string & text) {
        if (text.empty()) {
            return;
        }

        const std::vector<llama_token> tokens = common_tokenize(vocab, text, false, true);

        for (size_t i = 0; i + n < tokens.size(); ++i) {
            mod.add(tokens.data() + i);
        }

        n_docs   += 1;
        n_tokens += tokens.size();
        n_bytes  += text.size();
    };

    for (const std::string & input : inputs) {
        std::string data;
        if (input == "-") {
            data.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
        } else {
            std::ifstream f(input, std::ios::binary);
            if (!f) {
                fprintf(stderr, "cannot read %s\n", input.c_str());
                return 1;
            }
            data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }

        size_t pos = 0;
        while (pos <= data.size()) {
            size_t end = data.find('\0', pos);
            if (end == std::string::npos) {
                end = data.size();
            }
            add_doc(data.substr(pos, end - pos));
            pos = end + 1;
        }

        fprintf(stderr, "%s: %zu documents, %zu tokens so far, %zu/%zu cells used (%.1f%%)\n",
                input.c_str(), n_docs, n_tokens, mod.get_used(), mod.size(), 100.0*mod.get_used()/mod.size());
    }

    if (!mod.save(fname_out, n_vocab, err)) {
        fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    fprintf(stderr, "wrote %s: %zu documents, %.1f MiB of text, %zu tokens, %zu/%zu cells used (%.1f%%), n_match = %zu\n",
            fname_out.c_str(), n_docs, n_bytes/1024.0/1024.0, n_tokens, mod.get_used(), mod.size(), 100.0*mod.get_used()/mod.size(), n);

    if (mod.get_used() > mod.size()/2) {
        fprintf(stderr, "note: the table is more than half full, n-grams are overwriting each other - consider a larger --size\n");
    }

    llama_model_free(model);
    llama_backend_free();

    return 0;
}
