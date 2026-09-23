#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <string>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
// each cell keeps a 32-bit fingerprint of its n-gram, so a lookup that lands on a
// cell written by a different n-gram returns EMPTY instead of a wrong token
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    struct cell_t {
        uint32_t key;
        entry_t  token;
    };

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    // binary dump of the table, tagged with n and n_vocab so a table from another model is refused
    // load() takes the table size from the file
    bool save(const std::string & path, uint32_t n_vocab, std::string & err) const;
    bool load(const std::string & path, uint32_t n_vocab, std::string & err);

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    uint64_t hash(const entry_t * tokens) const;

    size_t n; // ngram size to hash

    size_t used;

    std::vector<cell_t> entries;
};
