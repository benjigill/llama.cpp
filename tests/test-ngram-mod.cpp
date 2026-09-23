#include "ngram-mod.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

using entry_t = common_ngram_mod::entry_t;

int main() {
    const size_t n = 4;

    // hits and misses
    {
        common_ngram_mod mod(n, 1 << 16);

        const std::vector<entry_t> a = { 1, 2, 3, 4, 5 };
        const std::vector<entry_t> b = { 9, 8, 7, 6, 5 };

        CHECK(mod.get(a.data()) == common_ngram_mod::EMPTY);

        mod.add(a.data());
        CHECK(mod.get(a.data()) == 5);
        CHECK(mod.get(b.data()) == common_ngram_mod::EMPTY);
        CHECK(mod.get_used() == 1);

        // same n-gram, new continuation: replaced, not counted twice
        const std::vector<entry_t> a2 = { 1, 2, 3, 4, 42 };
        mod.add(a2.data());
        CHECK(mod.get(a.data()) == 42);
        CHECK(mod.get_used() == 1);
    }

    // a one-bucket table: every n-gram collides, the fingerprint rejects the others
    {
        common_ngram_mod mod(n, 1);

        const std::vector<entry_t> a = { 1, 2, 3, 4, 5 };
        const std::vector<entry_t> b = { 9, 8, 7, 6, 11 };

        mod.add(a.data());
        CHECK(mod.get(b.data()) == common_ngram_mod::EMPTY);

        mod.add(b.data());
        CHECK(mod.get(b.data()) == 11);
        CHECK(mod.get(a.data()) == common_ngram_mod::EMPTY);
    }

    // save / load
    {
        const std::string path = "test-ngram-mod.bin";
        std::string err;

        common_ngram_mod mod(n, 1000);
        std::vector<entry_t> toks;
        for (int i = 0; i < 500; ++i) {
            toks.push_back((i*7919) % 1000);
        }
        for (size_t i = 0; i + n < toks.size(); ++i) {
            mod.add(toks.data() + i);
        }
        CHECK(mod.save(path, 1000, err));

        common_ngram_mod mod2(n, 16);
        CHECK(mod2.load(path, 1000, err));
        CHECK(mod2.size() == mod.size());
        CHECK(mod2.get_used() == mod.get_used());
        for (size_t i = 0; i + n < toks.size(); ++i) {
            CHECK(mod2.get(toks.data() + i) == mod.get(toks.data() + i));
        }

        // refused: other vocab, other n-gram length, not a table
        common_ngram_mod mod3(n, 16);
        CHECK(!mod3.load(path, 999, err));
        common_ngram_mod mod4(n + 1, 16);
        CHECK(!mod4.load(path, 1000, err));
        CHECK(mod3.size() == 16 && mod4.size() == 16);

        // truncated: refused, the table keeps its size and is empty
        {
            FILE * f = std::fopen(path.c_str(), "rb+");
            CHECK(f);
            std::fseek(f, 0, SEEK_END);
            const long len = std::ftell(f);
            std::fclose(f);

            std::vector<char> buf(len);
            f = std::fopen(path.c_str(), "rb");
            CHECK(std::fread(buf.data(), 1, len, f) == (size_t) len);
            std::fclose(f);
            f = std::fopen(path.c_str(), "wb");
            std::fwrite(buf.data(), 1, len - 8, f);
            std::fclose(f);

            common_ngram_mod mod5(n, 16);
            CHECK(!mod5.load(path, 1000, err));
            CHECK(mod5.size() == 16 && mod5.get_used() == 0);
            CHECK(mod5.get(toks.data()) == common_ngram_mod::EMPTY);
        }

        FILE * f = std::fopen(path.c_str(), "wb");
        std::fputs("not a table", f);
        std::fclose(f);
        CHECK(!mod3.load(path, 1000, err));

        std::remove(path.c_str());
    }

    fprintf(stderr, "OK\n");
    return 0;
}
