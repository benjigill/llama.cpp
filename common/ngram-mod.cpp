#include "ngram-mod.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

uint64_t common_ngram_mod::hash(const entry_t * tokens) const {
    uint64_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    return res;
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    return hash(tokens) % entries.size();
}

// fingerprint from the high bits of a remix of the hash, independent of the bucket index
static uint32_t common_ngram_mod_key(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return (uint32_t) (h >> 32);
}

void common_ngram_mod::add(const entry_t * tokens) {
    const uint64_t h = hash(tokens);
    cell_t & c = entries[h % entries.size()];

    if (c.token == EMPTY) {
        used++;
    }

    c = { common_ngram_mod_key(h), tokens[n] };
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const uint64_t h = hash(tokens);
    const cell_t & c = entries[h % entries.size()];

    return c.key == common_ngram_mod_key(h) ? c.token : EMPTY;
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), cell_t { 0, EMPTY });
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}

//
// file format: header, then size() cells
//

static constexpr char     NGRAM_MOD_MAGIC[8] = { 'N', 'G', 'R', 'A', 'M', 'M', 'O', 'D' };
static constexpr uint32_t NGRAM_MOD_VERSION  = 1;

struct common_ngram_mod_header {
    char     magic[8];
    uint32_t version;
    uint32_t n;
    uint32_t n_vocab;
    uint32_t cell_size;
    uint64_t size;
    uint64_t used;
};

bool common_ngram_mod::save(const std::string & path, uint32_t n_vocab, std::string & err) const {
    // write to a temporary file and rename, so an interrupted save keeps the previous table
    const std::string tmp = path + ".tmp";

    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "cannot open " + tmp + " for writing";
        return false;
    }

    common_ngram_mod_header hdr = {};
    std::memcpy(hdr.magic, NGRAM_MOD_MAGIC, sizeof(hdr.magic));
    hdr.version   = NGRAM_MOD_VERSION;
    hdr.n         = (uint32_t) n;
    hdr.n_vocab   = n_vocab;
    hdr.cell_size = sizeof(cell_t);
    hdr.size      = entries.size();
    hdr.used      = used;

    bool ok = std::fwrite(&hdr, sizeof(hdr), 1, f) == 1 &&
              std::fwrite(entries.data(), sizeof(cell_t), entries.size(), f) == entries.size();
    ok = std::fclose(f) == 0 && ok;

    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "failed to write " + path;
        return false;
    }

    return true;
}

bool common_ngram_mod::load(const std::string & path, uint32_t n_vocab, std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }

    common_ngram_mod_header hdr = {};

    bool ok = std::fread(&hdr, sizeof(hdr), 1, f) == 1;
    if (!ok || std::memcmp(hdr.magic, NGRAM_MOD_MAGIC, sizeof(hdr.magic)) != 0 || hdr.version != NGRAM_MOD_VERSION || hdr.cell_size != sizeof(cell_t)) {
        err = path + " is not an ngram-mod table (or has an unsupported version)";
    } else if (hdr.n != n) {
        err = path + " was built with n_match = " + std::to_string(hdr.n) + ", expected " + std::to_string(n);
    } else if (hdr.n_vocab != n_vocab) {
        err = path + " was built for a vocab of " + std::to_string(hdr.n_vocab) + " tokens, the model has " + std::to_string(n_vocab);
    } else if (hdr.size == 0) {
        err = path + " has an empty table";
    } else {
        // free the current table first, so a large table is not held twice
        const size_t size_old = entries.size();
        std::vector<cell_t>().swap(entries);
        entries.resize(hdr.size);

        if (std::fread(entries.data(), sizeof(cell_t), entries.size(), f) == entries.size()) {
            used = hdr.used;
            std::fclose(f);
            return true;
        }

        err = path + " is truncated";
        entries.assign(size_old, cell_t { 0, EMPTY });
        used = 0;
    }

    std::fclose(f);
    return false;
}
