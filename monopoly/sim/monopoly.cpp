// Monopoly strategy simulator: standard US board, official rules.
//
// Rules modelled: 40 squares, two dice, doubles roll again, third double goes to jail;
// both 16-card decks (current US text) drawn from the top and put back at the bottom,
// Get Out of Jail Free cards held until used; jail (pay $50, use a card, or roll for
// doubles for up to three turns, then pay); rent doubled on unimproved full colour sets;
// railroads 25/50/100/200, utilities 4x/10x; the two "nearest railroad" Chance cards pay
// double, the "nearest utility" card pays 10x a fresh roll; even building, a bank of
// 32 houses and 12 hotels; mortgages at half price, lifted at +10%; selling buildings
// at half cost; bankruptcy to a player (who takes everything) or to the bank (property
// goes back on sale, auctioned when auctions are on); auctions of unbought property;
// optional house rules: Free Parking jackpot, $400 for landing on Go, no auctions.
// Trading: two players swap the pieces that complete a colour set for each of them,
// with cash to balance, when both agree by their own valuation.
//
// Build: g++ -O2 -std=c++17 -pthread -o monopoly monopoly.cpp
// Usage: ./monopoly --strats "everything|railroads|darkblue|orange" --games 100000 [options]
//   --auctions 0/1 (default 1)  --trading 0/1 (default 0)  --fp 0/1  --go400 0/1
//   --maxrounds N (default 1000)  --seed S  --threads T  --give "0:orange,1:green"
//   --cash N (starting cash, default 1500)  --log file.jsonl (event log of game 0)
//   --freq N (print landing frequency from N single-token turns and exit)

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------- board
enum { T_PROP, T_RR, T_UTIL, T_GO, T_CC, T_CH, T_TAX, T_JAIL, T_FP, T_G2J };
struct Sq { const char* name; int type; int price; int group; int rent[6]; int hc; int tax; };
static const Sq B[40] = {
    {"Go", T_GO, 0, -1, {0}, 0, 0},
    {"Mediterranean Avenue", T_PROP, 60, 0, {2, 10, 30, 90, 160, 250}, 50, 0},
    {"Community Chest", T_CC, 0, -1, {0}, 0, 0},
    {"Baltic Avenue", T_PROP, 60, 0, {4, 20, 60, 180, 320, 450}, 50, 0},
    {"Income Tax", T_TAX, 0, -1, {0}, 0, 200},
    {"Reading Railroad", T_RR, 200, 8, {0}, 0, 0},
    {"Oriental Avenue", T_PROP, 100, 1, {6, 30, 90, 270, 400, 550}, 50, 0},
    {"Chance", T_CH, 0, -1, {0}, 0, 0},
    {"Vermont Avenue", T_PROP, 100, 1, {6, 30, 90, 270, 400, 550}, 50, 0},
    {"Connecticut Avenue", T_PROP, 120, 1, {8, 40, 100, 300, 450, 600}, 50, 0},
    {"Jail", T_JAIL, 0, -1, {0}, 0, 0},
    {"St. Charles Place", T_PROP, 140, 2, {10, 50, 150, 450, 625, 750}, 100, 0},
    {"Electric Company", T_UTIL, 150, 9, {0}, 0, 0},
    {"States Avenue", T_PROP, 140, 2, {10, 50, 150, 450, 625, 750}, 100, 0},
    {"Virginia Avenue", T_PROP, 160, 2, {12, 60, 180, 500, 700, 900}, 100, 0},
    {"Pennsylvania Railroad", T_RR, 200, 8, {0}, 0, 0},
    {"St. James Place", T_PROP, 180, 3, {14, 70, 200, 550, 750, 950}, 100, 0},
    {"Community Chest", T_CC, 0, -1, {0}, 0, 0},
    {"Tennessee Avenue", T_PROP, 180, 3, {14, 70, 200, 550, 750, 950}, 100, 0},
    {"New York Avenue", T_PROP, 200, 3, {16, 80, 220, 600, 800, 1000}, 100, 0},
    {"Free Parking", T_FP, 0, -1, {0}, 0, 0},
    {"Kentucky Avenue", T_PROP, 220, 4, {18, 90, 250, 700, 875, 1050}, 150, 0},
    {"Chance", T_CH, 0, -1, {0}, 0, 0},
    {"Indiana Avenue", T_PROP, 220, 4, {18, 90, 250, 700, 875, 1050}, 150, 0},
    {"Illinois Avenue", T_PROP, 240, 4, {20, 100, 300, 750, 925, 1100}, 150, 0},
    {"B. & O. Railroad", T_RR, 200, 8, {0}, 0, 0},
    {"Atlantic Avenue", T_PROP, 260, 5, {22, 110, 330, 800, 975, 1150}, 150, 0},
    {"Ventnor Avenue", T_PROP, 260, 5, {22, 110, 330, 800, 975, 1150}, 150, 0},
    {"Water Works", T_UTIL, 150, 9, {0}, 0, 0},
    {"Marvin Gardens", T_PROP, 280, 5, {24, 120, 360, 850, 1025, 1200}, 150, 0},
    {"Go To Jail", T_G2J, 0, -1, {0}, 0, 0},
    {"Pacific Avenue", T_PROP, 300, 6, {26, 130, 390, 900, 1100, 1275}, 200, 0},
    {"North Carolina Avenue", T_PROP, 300, 6, {26, 130, 390, 900, 1100, 1275}, 200, 0},
    {"Community Chest", T_CC, 0, -1, {0}, 0, 0},
    {"Pennsylvania Avenue", T_PROP, 320, 6, {28, 150, 450, 1000, 1200, 1400}, 200, 0},
    {"Short Line", T_RR, 200, 8, {0}, 0, 0},
    {"Chance", T_CH, 0, -1, {0}, 0, 0},
    {"Park Place", T_PROP, 350, 7, {35, 175, 500, 1100, 1300, 1500}, 200, 0},
    {"Luxury Tax", T_TAX, 0, -1, {0}, 0, 100},
    {"Boardwalk", T_PROP, 400, 7, {50, 200, 600, 1400, 1700, 2000}, 200, 0},
};
static const char* GROUP_NAME[10] = {"brown", "lightblue", "pink", "orange", "red",
                                     "yellow", "green", "darkblue", "railroads", "utilities"};
static int GSZ[10], GSQ[10][4];
static double FREQ[40];        // landings per turn of one token (estimated at start-up)
static double TV[8];           // "true value" of completing a colour set, used by shark traders
static int BUILD_ORDER[8];     // colour sets by return on a 3-house build, best first

static inline bool buyable(int s) { int t = B[s].type; return t == T_PROP || t == T_RR || t == T_UTIL; }

// ---------------------------------------------------------------- rng
struct Rng {
    uint64_t s[4];
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    void seed(uint64_t x) { for (auto& v : s) v = splitmix(x); }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    inline int below(int n) { return (int)(((next() >> 32) * (uint64_t)n) >> 32); }
    inline int die() { return below(6) + 1; }
};

// ---------------------------------------------------------------- strategy
struct Strat {
    std::string name;
    uint64_t buy = 0;        // bitmask of squares this player buys when landing (and bids on)
    int reserve = 0;         // cash kept back when buying or bidding
    double bid = 1.0;        // max auction bid, multiple of list price, on wanted squares
    double setBid = 1.0;     // extra multiplier when the square completes own set
    double blockBid = 0.0;   // bid this multiple of price to stop an opponent completing a set
    int buildMax = 5;        // 0 none, 1..4 houses, 5 hotel
    int buildReserve = 0;    // cash kept back when building
    int bpri = 0;            // 0: finish one set before the next; 1: level by level across sets
    int jail = 0;            // 0 leave at once, 1 stay as long as allowed, 2 leave early, stay late
    int trade = 1;           // 0 refuses trades, 1 fair (list price), 2 shark (set value)
    bool m2b = false;        // mortgage loose properties to pay for houses
    bool stretch = false;    // ignore reserve to buy a square that completes own set
    double sellPremium = 2.5; // cash buyouts: sells a set-completing piece for this multiple of list price
    double buyPremium = 3.0;  // cash buyouts: pays up to this multiple of list price to complete a set
    // a favourite (a fan still buys everything, but leans toward these squares)
    uint64_t fav = 0;         // favourite squares
    double favBid = 1.0;      // bids up to this multiple of list price for a favourite at auction
    bool favMortgage = false; // mortgages other property to afford a favourite it lands on
    int holdFor = 0;          // keeps this much cash back on other purchases while a favourite is unsold
    // building choices
    int firstGroup = -1;      // build this colour set before any other
    int border = 0;           // order of the rest: 0 by return on 3 houses, 1 cheapest houses first, 2 dearest first
    double dangerFrac = 0;    // when building, keep this multiple of the biggest rent an opponent can charge now
    // other decisions a player controls
    int buyMort = 0;          // mortgage other property to buy what you land on: 1 if it completes or blocks a set, 2 always
    int tradeCash = 0;        // only make a swap when holding at least this much cash (to build straight away)
    bool rrTrade = false;     // also swap set-for-set to collect all four railroads
    int raiseOrder = 0;       // short of cash: 0 mortgage loose property before selling houses, 1 sell houses first
};

static uint64_t maskFromName(const std::string& w) {
    uint64_t m = 0;
    if (w == "all") { for (int s = 0; s < 40; s++) if (buyable(s)) m |= 1ull << s; return m; }
    if (w == "none") return 0;
    for (int g = 0; g < 10; g++) {
        std::string gn = GROUP_NAME[g];
        if (w == gn || (w == "rr" && g == 8) || (w == "util" && g == 9) || (w == "colors" && g < 8)) {
            for (int i = 0; i < GSZ[g]; i++) m |= 1ull << GSQ[g][i];
        }
    }
    if (m == 0 && !w.empty() && isdigit((unsigned char)w[0])) m |= 1ull << atoi(w.c_str());
    if (m == 0 && w != "colors") { fprintf(stderr, "unknown buy set '%s'\n", w.c_str()); exit(1); }
    return m;
}
static uint64_t parseMask(const std::string& v) {
    uint64_t m = 0; std::string cur;
    bool minus = false; std::stringstream ss(v); std::string tok;
    // "all-rr-util" removes, "orange+red" adds
    size_t i = 0; std::string acc; char op = '+';
    auto flush = [&](char o, const std::string& w) { if (w.empty()) return; uint64_t x = maskFromName(w); if (o == '+') m |= x; else m &= ~x; };
    for (; i <= v.size(); i++) {
        if (i == v.size() || v[i] == '+' || v[i] == '-') { flush(op, acc); acc.clear(); if (i < v.size()) op = v[i]; }
        else acc += v[i];
    }
    (void)minus; (void)cur; (void)tok;
    return m;
}

static Strat presetStrat(const std::string& n) {
    Strat s; s.name = n; s.buy = maskFromName("all");
    if (n == "everything") {}
    else if (n == "nothing") { s.buy = 0; }
    else if (n == "railroads") { s.buy = maskFromName("rr"); }
    else if (n == "utilities") { s.buy = maskFromName("utilities"); }
    else if (n == "infrastructure") { s.buy = maskFromName("rr") | maskFromName("utilities"); }
    else if (n == "boardwalk" || n == "darkblue") { s.buy = maskFromName("darkblue"); }
    else if (n == "colors") { s.buy = maskFromName("colors"); }
    else if (n == "threehouses") { s.buildMax = 3; }
    else if (n == "houselock") { s.buildMax = 4; }
    else if (n == "jailbird") { s.jail = 1; }
    else if (n == "smartjail") { s.jail = 2; }
    else if (n == "prudent") { s.reserve = 300; s.buildReserve = 300; }
    else if (n == "stubborn") { s.trade = 0; }
    else if (n == "shark") { s.trade = 2; }
    else if (n == "allin") { s.m2b = true; s.jail = 2; }
    else if (n == "champion") {
        s.bid = 1.1; s.setBid = 1.8; s.blockBid = 1.4; s.buildMax = 5; s.bpri = 1; s.buildReserve = 100;
        s.jail = 2; s.trade = 2; s.m2b = true; s.stretch = true;
    } else {
        bool found = false;
        for (int g = 0; g < 10; g++) if (n == GROUP_NAME[g]) { s.buy = maskFromName(n); found = true; }
        if (!found) { fprintf(stderr, "unknown strategy '%s'\n", n.c_str()); exit(1); }
    }
    return s;
}

// spec: "preset" or "preset;key=value;key=value" or "name=X;buy=...;..."
static Strat parseStrat(const std::string& spec) {
    std::vector<std::string> parts; std::stringstream ss(spec); std::string p;
    while (std::getline(ss, p, ';')) if (!p.empty()) parts.push_back(p);
    Strat s; size_t start = 0;
    if (!parts.empty() && parts[0].find('=') == std::string::npos) { s = presetStrat(parts[0]); start = 1; }
    else s = presetStrat("everything");
    for (size_t i = start; i < parts.size(); i++) {
        auto eq = parts[i].find('='); std::string k = parts[i].substr(0, eq), v = parts[i].substr(eq + 1);
        if (k == "name") s.name = v;
        else if (k == "buy") s.buy = parseMask(v);
        else if (k == "reserve") s.reserve = atoi(v.c_str());
        else if (k == "bid") s.bid = atof(v.c_str());
        else if (k == "setbid") s.setBid = atof(v.c_str());
        else if (k == "blockbid") s.blockBid = atof(v.c_str());
        else if (k == "build") s.buildMax = atoi(v.c_str());
        else if (k == "breserve") s.buildReserve = atoi(v.c_str());
        else if (k == "bpri") s.bpri = (v == "priority" ? 0 : v == "spread" ? 1 : v == "focus" ? 2 : v == "staged" ? 3 : atoi(v.c_str()));
        else if (k == "jail") s.jail = (v == "leave" ? 0 : v == "stay" ? 1 : v == "smart" ? 2 : v == "zone" ? 3 : v == "houses" ? 4 :
                                        v == "unsold" ? 5 : v == "rollfirst" ? 6 : atoi(v.c_str()));
        else if (k == "trade") s.trade = (v == "none" ? 0 : v == "fair" ? 1 : v == "shark" ? 2 : v == "picky" ? 3 : atoi(v.c_str()));
        else if (k == "m2b") s.m2b = atoi(v.c_str()) != 0;
        else if (k == "stretch") s.stretch = atoi(v.c_str()) != 0;
        else if (k == "sell") s.sellPremium = atof(v.c_str());
        else if (k == "buyp") s.buyPremium = atof(v.c_str());
        else if (k == "fav") { s.fav = parseMask(v); s.favBid = 2.0; s.favMortgage = true; }
        else if (k == "favbid") s.favBid = atof(v.c_str());
        else if (k == "favmort") s.favMortgage = atoi(v.c_str()) != 0;
        else if (k == "hold") s.holdFor = atoi(v.c_str());
        else if (k == "first") { s.firstGroup = -1; for (int g = 0; g < 8; g++) if (v == GROUP_NAME[g]) s.firstGroup = g; }
        else if (k == "border") s.border = (v == "roi" ? 0 : v == "cheap" ? 1 : v == "dear" ? 2 : atoi(v.c_str()));
        else if (k == "danger") s.dangerFrac = atof(v.c_str());
        else if (k == "buymort") s.buyMort = atoi(v.c_str());
        else if (k == "tcash") s.tradeCash = atoi(v.c_str());
        else if (k == "rrtrade") s.rrTrade = atoi(v.c_str()) != 0;
        else if (k == "raise") s.raiseOrder = (v == "mortgage" ? 0 : v == "sell" ? 1 : atoi(v.c_str()));
        else { fprintf(stderr, "unknown key '%s'\n", k.c_str()); exit(1); }
    }
    return s;
}

// ---------------------------------------------------------------- config & stats
struct Config {
    bool auctions = true, trading = false, freeParking = false, go400 = false, buyouts = false, deal = false;
    int maxRounds = 1000, startCash = 1500;
    int endgame = 0;  // >0: once nearly everything is owned and this many rounds pass with no new set, trade or bankruptcy,
                      // players start selling set cards for cash and buying up cards split between players

    std::vector<std::pair<int, int>> give;  // (slot, group) handed out before the first roll
};

struct Stats {
    int n = 0;                          // players
    long games = 0, timeouts = 0;
    std::vector<long> wins, nwWins, seatWins, seatGames, gotMono, winWithMono, firstOut, busted, allRR, allRRWin;
    std::vector<std::array<long, 10>> monoBy, winMonoBy;   // [slot][group]
    std::vector<double> netAtEnd, rent;
    std::array<double, 40> rentSq{}, landSq{};
    std::array<long, 10> winnerFirstMono{}, firstMonoAny{};
    long trades = 0, endgameDeals = 0;
    double roundsSum = 0; std::vector<long> roundsHist;      // finished games, by 10-round bins
    long turnsTotal = 0;
    void init(int np) {
        n = np; wins.assign(n, 0); nwWins.assign(n, 0); seatWins.assign(n, 0); seatGames.assign(n, 0);
        gotMono.assign(n, 0); winWithMono.assign(n, 0); monoBy.assign(n, {}); winMonoBy.assign(n, {});
        firstOut.assign(n, 0); busted.assign(n, 0); allRR.assign(n, 0); allRRWin.assign(n, 0);
        netAtEnd.assign(n, 0); rent.assign(n, 0); roundsHist.assign(101, 0);
    }
    void add(const Stats& o) {
        games += o.games; timeouts += o.timeouts; trades += o.trades; endgameDeals += o.endgameDeals; roundsSum += o.roundsSum; turnsTotal += o.turnsTotal;
        for (int i = 0; i < n; i++) {
            wins[i] += o.wins[i]; nwWins[i] += o.nwWins[i]; seatWins[i] += o.seatWins[i]; seatGames[i] += o.seatGames[i];
            gotMono[i] += o.gotMono[i]; winWithMono[i] += o.winWithMono[i]; netAtEnd[i] += o.netAtEnd[i]; rent[i] += o.rent[i];
            firstOut[i] += o.firstOut[i]; busted[i] += o.busted[i]; allRR[i] += o.allRR[i]; allRRWin[i] += o.allRRWin[i];
            for (int g = 0; g < 10; g++) { monoBy[i][g] += o.monoBy[i][g]; winMonoBy[i][g] += o.winMonoBy[i][g]; }
        }
        for (int s = 0; s < 40; s++) { rentSq[s] += o.rentSq[s]; landSq[s] += o.landSq[s]; }
        for (int g = 0; g < 10; g++) { winnerFirstMono[g] += o.winnerFirstMono[g]; firstMonoAny[g] += o.firstMonoAny[g]; }
        for (size_t i = 0; i < roundsHist.size(); i++) roundsHist[i] += o.roundsHist[i];
    }
};

// ---------------------------------------------------------------- game
enum { CH_BOARDWALK, CH_GO, CH_ILLINOIS, CH_STCHARLES, CH_RR1, CH_RR2, CH_UTIL, CH_DIV50, CH_GOOJF, CH_BACK3,
       CH_JAIL, CH_REPAIRS, CH_SPEED15, CH_READING, CH_CHAIRMAN, CH_LOAN150 };
enum { CC_GO, CC_BANKERR200, CC_DOCTOR50, CC_STOCK50, CC_GOOJF, CC_JAIL, CC_HOLIDAY100, CC_TAXREF20, CC_BIRTHDAY10,
       CC_LIFE100, CC_HOSPITAL100, CC_SCHOOL50, CC_CONSULT25, CC_REPAIRS, CC_BEAUTY10, CC_INHERIT100 };

struct Deck {
    int c[16]; int head = 0, cnt = 16;
    void shuffle(Rng& r) { for (int i = 0; i < 16; i++) c[i] = i; for (int i = 15; i > 0; i--) std::swap(c[i], c[r.below(i + 1)]); head = 0; cnt = 16; }
    int draw() { int x = c[head]; head = (head + 1) & 15; cnt--; return x; }
    void bottom(int x) { c[(head + cnt) & 15] = x; cnt++; }
};

struct Player {
    int cash = 0, pos = 0, jailTurns = 0, slot = 0;
    bool inJail = false, alive = true;
    int goojfCh = 0, goojfCc = 0;
    int monoMask = 0, firstMono = -1; long rent = 0; bool rrAll = false;
};

static const int BANK = -1, POT = -2;

struct Game {
    const Config& cfg; Rng rng; int n; std::vector<const Strat*> S; Player P[8];
    int owner[40], houses[40]; bool mort[40];
    int bankHouses = 32, bankHotels = 12, pot = 0, round = 0, aliveCount = 0;
    Deck chance, chest; Stats* st; FILE* log = nullptr; long turns = 0; bool transferring = false; long lastProgress = 0;

    Game(const Config& c, uint64_t seed, std::vector<const Strat*> strats, Stats* stats)
        : cfg(c), n((int)strats.size()), S(strats), st(stats) { rng.seed(seed); }

    // -------- helpers
    int countOwned(int p, int g) const { int k = 0; for (int i = 0; i < GSZ[g]; i++) k += owner[GSQ[g][i]] == p; return k; }
    bool hasSet(int p, int g) const { return countOwned(p, g) == GSZ[g]; }
    bool groupHasBuildings(int g) const { for (int i = 0; i < GSZ[g]; i++) if (houses[GSQ[g][i]]) return true; return false; }
    bool completesSet(int p, int s) const {
        int g = B[s].group; if (g < 0 || g > 7) return false;
        for (int i = 0; i < GSZ[g]; i++) { int q = GSQ[g][i]; if (q != s && owner[q] != p) return false; }
        return true;
    }
    bool completesOpp(int p, int s) const {
        for (int q = 0; q < n; q++) if (q != p && P[q].alive && completesSet(q, s)) return true;
        return false;
    }
    int netWorth(int p) const {
        int w = P[p].cash;
        for (int s = 0; s < 40; s++) if (owner[s] == p) {
            w += mort[s] ? B[s].price / 2 : B[s].price;
            w += (houses[s] == 5 ? 5 : houses[s]) * B[s].hc;
        }
        return w;
    }
    void updateMono(int p) {
        if (!P[p].rrAll && !transferring && hasSet(p, 8)) P[p].rrAll = true;  // collected, not inherited from a bankrupt
        for (int g = 0; g < 8; g++) if (!(P[p].monoMask >> g & 1) && hasSet(p, g)) {
            P[p].monoMask |= 1 << g;
            lastProgress = turns;
            if (P[p].firstMono < 0) P[p].firstMono = g;
            if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"set\",\"g\":\"%s\"}\n", round, p, GROUP_NAME[g]);
        }
    }
    bool anyOppDeveloped(int p) const {
        for (int s = 0; s < 40; s++) if (owner[s] >= 0 && owner[s] != p && houses[s] > 0) return true;
        return false;
    }
    int unownedCount() const { int k = 0; for (int s = 0; s < 40; s++) if (buyable(s) && owner[s] < 0) k++; return k; }

    // -------- money
    void receive(int to, int amt) { if (to >= 0) P[to].cash += amt; else if (to == POT && cfg.freeParking) pot += amt; }

    bool mortgageOne(int p, int pass, int skipGroup = -1) {
        // pass 0: loose property (no full set), utilities first, then colours by price, railroads last
        // pass 1: properties of a full set with no buildings
        int best = -1, bestKey = 1 << 30;
        for (int s = 0; s < 40; s++) {
            if (owner[s] != p || mort[s] || houses[s]) continue;
            int g = B[s].group; bool inSet = g < 8 && hasSet(p, g);
            if (g == skipGroup) continue;
            if (inSet && groupHasBuildings(g)) continue;
            if ((pass == 0) == inSet) continue;
            int key = B[s].type == T_UTIL ? 0 : B[s].type == T_RR ? 10000 : B[s].price;
            if (key < bestKey) { bestKey = key; best = s; }
        }
        if (best < 0) return false;
        mort[best] = true; P[p].cash += B[best].price / 2;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"mortgage\",\"s\":%d}\n", round, p, best);
        return true;
    }
    bool sellOneLevel(int p) {
        // sell from the least valuable developed set first, from the most-built lot (even rule)
        for (int k = 7; k >= 0; k--) {
            int g = BUILD_ORDER[k];
            if (!hasSet(p, g) || !groupHasBuildings(g)) continue;
            int s = -1, mx = -1;
            for (int i = 0; i < GSZ[g]; i++) { int q = GSQ[g][i]; if (houses[q] > mx) { mx = houses[q]; s = q; } }
            if (houses[s] == 5) {
                if (bankHouses >= 4) { houses[s] = 4; bankHotels++; bankHouses -= 4; P[p].cash += B[s].hc / 2; }
                else {  // not enough houses to break the hotel down: sell the whole group's buildings
                    for (int i = 0; i < GSZ[g]; i++) { int q = GSQ[g][i];
                        if (houses[q] == 5) { bankHotels++; P[p].cash += B[q].hc * 5 / 2; }
                        else { bankHouses += houses[q]; P[p].cash += B[q].hc * houses[q] / 2; }
                        houses[q] = 0; }
                }
            } else { houses[s]--; bankHouses++; P[p].cash += B[s].hc / 2; }
            return true;
        }
        return false;
    }
    bool raise(int p, int need) {
        while (P[p].cash < need) {
            if (S[p]->raiseOrder == 1 && sellOneLevel(p)) continue;
            if (mortgageOne(p, 0)) continue;
            if (sellOneLevel(p)) continue;
            if (mortgageOne(p, 1)) continue;
            return false;
        }
        return true;
    }
    // returns false if the payer went bankrupt
    bool pay(int from, int to, int amt) {
        if (amt <= 0) return true;
        if (P[from].cash < amt && !raise(from, amt)) {
            int got = std::max(0, P[from].cash);
            bankrupt(from, to >= 0 ? to : BANK);
            (void)got;
            return false;
        }
        P[from].cash -= amt; receive(to, amt);
        return true;
    }
    void bankrupt(int p, int cred) {
        lastProgress = turns;
        if (aliveCount == n) st->firstOut[p]++;
        st->busted[p]++;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"bankrupt\",\"to\":%d}\n", round, p, cred);
        for (int s = 0; s < 40; s++) if (owner[s] == p && houses[s]) {
            if (houses[s] == 5) { bankHotels++; P[p].cash += B[s].hc * 5 / 2; }
            else { bankHouses += houses[s]; P[p].cash += B[s].hc * houses[s] / 2; }
            houses[s] = 0;
        }
        std::vector<int> back;
        if (cred >= 0) {
            P[cred].cash += std::max(0, P[p].cash);
            for (int s = 0; s < 40; s++) if (owner[s] == p) {
                owner[s] = cred;
                if (mort[s]) { int fee = B[s].price / 20; P[cred].cash -= std::min(fee, std::max(0, P[cred].cash)); }
            }
            P[cred].goojfCh += P[p].goojfCh; P[cred].goojfCc += P[p].goojfCc;
            transferring = true; updateMono(cred); transferring = false;
        } else {
            for (int s = 0; s < 40; s++) if (owner[s] == p) { owner[s] = -1; mort[s] = false; back.push_back(s); }
            for (int i = 0; i < P[p].goojfCh; i++) chance.bottom(CH_GOOJF);
            for (int i = 0; i < P[p].goojfCc; i++) chest.bottom(CC_GOOJF);
        }
        P[p].goojfCh = P[p].goojfCc = 0; P[p].cash = 0; P[p].alive = false; P[p].inJail = false; aliveCount--;
        if (cfg.auctions && aliveCount > 1) for (int s : back) auction(s, -1);
    }

    // -------- buying
    bool favOpen(int p) const {
        uint64_t f = S[p]->fav;
        for (int s = 0; f && s < 40; s++) if ((f >> s & 1) && owner[s] < 0) return true;
        return false;
    }
    int reserveFor(int p, int s) const {
        const Strat& t = *S[p];
        if (t.fav >> s & 1) return 0;                        // a fan always finds the money for a favourite
        if (t.stretch && completesSet(p, s)) return 0;
        return std::max(t.reserve, (t.holdFor && favOpen(p)) ? t.holdFor : 0);
    }
    int maxBid(int q, int s) const {
        const Strat& t = *S[q]; double f = 0;
        if (t.buy >> s & 1) { f = t.bid; if (completesSet(q, s)) f *= t.setBid; }
        if (t.fav >> s & 1) f = std::max(f, t.favBid);
        if (t.blockBid > 0 && completesOpp(q, s)) f = std::max(f, t.blockBid);
        if (f <= 0) return 0;
        int reserve = reserveFor(q, s);
        int b = std::min((int)(B[s].price * f), P[q].cash - reserve);
        return std::max(0, b);
    }
    void auction(int s, int decliner) {
        (void)decliner;
        int bids[8], bestBid = 0, second = 0;
        for (int q = 0; q < n; q++) bids[q] = P[q].alive ? maxBid(q, s) : 0;
        for (int q = 0; q < n; q++) bestBid = std::max(bestBid, bids[q]);
        if (bestBid < 10) return;
        int tied[8], nt = 0;
        for (int q = 0; q < n; q++) if (bids[q] == bestBid) tied[nt++] = q;
        int best = tied[nt > 1 ? rng.below(nt) : 0];
        for (int q = 0; q < n; q++) if (q != best) second = std::max(second, bids[q]);
        int price = std::min(bestBid, std::max(10, second + 1));
        P[best].cash -= price; owner[s] = best;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"auction\",\"s\":%d,\"price\":%d}\n", round, best, s, price);
        updateMono(best);
    }
    void offer(int p, int s) {
        const Strat& t = *S[p];
        bool want = t.buy >> s & 1;
        int reserve = reserveFor(p, s);
        if (want && t.favMortgage && (t.fav >> s & 1))
            while (P[p].cash < B[s].price && mortgageOne(p, 0)) {}
        if (want && t.buyMort && P[p].cash - B[s].price < reserve &&
            (t.buyMort == 2 || completesSet(p, s) || completesOpp(p, s) || (B[s].type == T_RR && t.rrTrade)))
            while (P[p].cash < B[s].price && mortgageOne(p, 0, B[s].group)) {}
        if (want && t.buyMort && P[p].cash >= B[s].price && P[p].cash - B[s].price < reserve &&
            (t.buyMort == 2 || completesSet(p, s) || completesOpp(p, s))) reserve = 0;
        if (want && P[p].cash - B[s].price >= reserve) {
            P[p].cash -= B[s].price; owner[s] = p;
            if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"buy\",\"s\":%d,\"price\":%d}\n", round, p, s, B[s].price);
            updateMono(p);
        } else if (cfg.auctions) auction(s, p);
    }

    // -------- rent
    int rentFor(int s, int dice, int cardMode) const {
        int o = owner[s];
        if (B[s].type == T_RR) {
            int k = 0; for (int i = 0; i < 4; i++) k += owner[GSQ[8][i]] == o;
            int r = 25 << (k - 1);
            return cardMode == 1 ? 2 * r : r;
        }
        if (B[s].type == T_UTIL) {
            if (cardMode == 2) return 10 * dice;
            int k = (owner[12] == o) + (owner[28] == o);
            return (k == 2 ? 10 : 4) * dice;
        }
        int h = houses[s];
        if (h) return B[s].rent[h];
        int r = B[s].rent[0];
        if (hasSet(o, B[s].group)) r *= 2;
        return r;
    }

    // -------- movement
    void goJail(int p) {
        P[p].pos = 10; P[p].inJail = true; P[p].jailTurns = 0;
        st->landSq[10] += 1;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"jail\"}\n", round, p);
    }
    void passGo(int p, int to) {
        if (cfg.go400 && to == 0) P[p].cash += 400; else P[p].cash += 200;
    }
    void advanceTo(int p, int to, int dice, int cardMode) {
        if (to < P[p].pos) passGo(p, to);
        P[p].pos = to; land(p, dice, cardMode);
    }
    void moveBy(int p, int k, int dice) {
        int to = P[p].pos + k;
        if (to >= 40) { to -= 40; passGo(p, to); }
        P[p].pos = to; land(p, dice, 0);
    }
    void land(int p, int dice, int cardMode) {
        int s = P[p].pos;
        st->landSq[s] += 1;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"land\",\"s\":%d,\"cash\":%d}\n", round, p, s, P[p].cash);
        switch (B[s].type) {
            case T_PROP: case T_RR: case T_UTIL: {
                int o = owner[s];
                if (o < 0) { offer(p, s); return; }
                if (o == p || mort[s]) return;
                int d = dice;
                if (cardMode == 2) d = rng.die() + rng.die();
                int r = rentFor(s, d, cardMode);
                int before = P[p].cash;
                bool ok = pay(p, o, r);
                int paid = ok ? r : std::max(0, before);
                st->rentSq[s] += paid; P[o].rent += paid;
                if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"rent\",\"s\":%d,\"to\":%d,\"amt\":%d}\n", round, p, s, o, r);
                return;
            }
            case T_TAX: pay(p, POT, B[s].tax); return;
            case T_G2J: goJail(p); return;
            case T_FP: if (cfg.freeParking && pot) { P[p].cash += pot; pot = 0; } return;
            case T_CH: drawChance(p, dice); return;
            case T_CC: drawChest(p, dice); return;
            default: return;
        }
    }
    int repairs(int p, int perHouse, int perHotel) const {
        int c = 0; for (int s = 0; s < 40; s++) if (owner[s] == p) c += houses[s] == 5 ? perHotel : houses[s] * perHouse;
        return c;
    }
    void drawChance(int p, int dice) {
        int c = chance.draw();
        if (c == CH_GOOJF) { P[p].goojfCh++; return; }
        chance.bottom(c);
        int pos = P[p].pos;
        switch (c) {
            case CH_BOARDWALK: advanceTo(p, 39, dice, 0); break;
            case CH_GO: advanceTo(p, 0, dice, 0); break;
            case CH_ILLINOIS: advanceTo(p, 24, dice, 0); break;
            case CH_STCHARLES: advanceTo(p, 11, dice, 0); break;
            case CH_RR1: case CH_RR2: { int to = pos < 5 ? 5 : pos < 15 ? 15 : pos < 25 ? 25 : pos < 35 ? 35 : 5; advanceTo(p, to, dice, 1); break; }
            case CH_UTIL: { int to = (pos < 12 || pos >= 28) ? 12 : 28; advanceTo(p, to, dice, 2); break; }
            case CH_DIV50: P[p].cash += 50; break;
            case CH_BACK3: P[p].pos = (pos + 37) % 40; land(p, dice, 0); break;
            case CH_JAIL: goJail(p); break;
            case CH_REPAIRS: pay(p, POT, repairs(p, 25, 100)); break;
            case CH_SPEED15: pay(p, POT, 15); break;
            case CH_READING: advanceTo(p, 5, dice, 0); break;
            case CH_CHAIRMAN: for (int q = 0; q < n; q++) if (q != p && P[q].alive && P[p].alive) pay(p, q, 50); break;
            case CH_LOAN150: P[p].cash += 150; break;
        }
    }
    void drawChest(int p, int dice) {
        int c = chest.draw();
        if (c == CC_GOOJF) { P[p].goojfCc++; return; }
        chest.bottom(c);
        switch (c) {
            case CC_GO: advanceTo(p, 0, dice, 0); break;
            case CC_BANKERR200: P[p].cash += 200; break;
            case CC_DOCTOR50: pay(p, POT, 50); break;
            case CC_STOCK50: P[p].cash += 50; break;
            case CC_JAIL: goJail(p); break;
            case CC_HOLIDAY100: case CC_LIFE100: case CC_INHERIT100: P[p].cash += 100; break;
            case CC_TAXREF20: P[p].cash += 20; break;
            case CC_BIRTHDAY10: for (int q = 0; q < n; q++) if (q != p && P[q].alive) pay(q, p, 10); break;
            case CC_HOSPITAL100: pay(p, POT, 100); break;
            case CC_SCHOOL50: pay(p, POT, 50); break;
            case CC_CONSULT25: P[p].cash += 25; break;
            case CC_REPAIRS: pay(p, POT, repairs(p, 40, 115)); break;
            case CC_BEAUTY10: P[p].cash += 10; break;
        }
    }

    // -------- building & mortgages
    bool canBuildOn(int p, int g) const {
        if (!hasSet(p, g)) return false;
        for (int i = 0; i < GSZ[g]; i++) if (mort[GSQ[g][i]]) return false;
        return true;
    }
    // the biggest rent any opponent could charge this player right now (utilities at an average roll of 7)
    int maxOppRent(int p) const {
        int m = 0;
        for (int s = 0; s < 40; s++) {
            int o = owner[s];
            if (o < 0 || o == p || mort[s] || !P[o].alive) continue;
            m = std::max(m, rentFor(s, 7, 0));
        }
        return m;
    }
    // build one level on group g if allowed; returns true if a house/hotel went up
    bool buildStep(int p, int g, int maxLevel, int reserve) {
        int s = -1, mn = 9;
        for (int i = 0; i < GSZ[g]; i++) { int q = GSQ[g][i]; if (houses[q] < mn) { mn = houses[q]; s = q; } }
        if (mn >= maxLevel) return false;
        int cost = B[s].hc;
        if (P[p].cash - cost < reserve) {
            if (!S[p]->m2b) return false;
            while (P[p].cash - cost < reserve && mortgageOne(p, 0)) {}
            if (P[p].cash - cost < reserve) return false;
        }
        if (mn == 4) { if (bankHotels == 0) return false; bankHotels--; bankHouses += 4; houses[s] = 5; }
        else { if (bankHouses == 0) return false; bankHouses--; houses[s]++; }
        P[p].cash -= cost;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"build\",\"s\":%d,\"level\":%d}\n", round, p, s, houses[s]);
        return true;
    }
    void manage(int p) {
        const Strat& t = *S[p];
        // lift mortgages on full sets first (needed before building), then on the rest
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < 10; k++) {
                int g = k < 8 ? BUILD_ORDER[k] : k;
                bool full = g < 8 && hasSet(p, g);
                if ((pass == 0) != full) continue;
                for (int i = 0; i < GSZ[g]; i++) {
                    int s = GSQ[g][i];
                    if (owner[s] != p || !mort[s]) continue;
                    int cost = B[s].price / 2 + B[s].price / 20;
                    int keep = pass == 0 ? t.buildReserve : std::max(t.reserve, t.buildReserve) + (t.m2b ? 600 : 150);
                    if (P[p].cash - cost >= keep) { P[p].cash -= cost; mort[s] = false; }
                }
            }
        if (t.buildMax <= 0) return;
        int reserve = t.buildReserve;
        if (t.dangerFrac > 0) reserve = std::max(reserve, (int)(t.dangerFrac * maxOppRent(p)));
        // build order: a named first set, then favourites, then by the chosen rule
        int base[8];
        for (int k = 0; k < 8; k++) base[k] = BUILD_ORDER[k];
        if (t.border) std::sort(base, base + 8, [&](int a, int b) {
            int ca = B[GSQ[a][0]].hc * 100 + B[GSQ[a][GSZ[a] - 1]].price, cb = B[GSQ[b][0]].hc * 100 + B[GSQ[b][GSZ[b] - 1]].price;
            return t.border == 1 ? ca < cb : ca > cb;
        });
        int order[8], no = 0;
        if (t.firstGroup >= 0) order[no++] = t.firstGroup;
        for (int k = 0; k < 8; k++) { int g = base[k]; if (g != t.firstGroup && (t.fav >> GSQ[g][0] & 1)) order[no++] = g; }
        for (int k = 0; k < 8; k++) { int g = base[k]; if (g != t.firstGroup && !(t.fav >> GSQ[g][0] & 1)) order[no++] = g; }
        if (t.bpri == 1) {          // spread: one level at a time across every set
            for (int L = 1; L <= t.buildMax; L++)
                for (int k = 0; k < 8; k++) { int g = order[k]; if (!canBuildOn(p, g)) continue; while (buildStep(p, g, L, reserve)) {} }
        } else if (t.bpri == 3) {   // staged: every set to three houses (in order), then on to hotels
            for (int L : {3, 5}) {
                int lv = std::min(L, (int)t.buildMax);
                for (int k = 0; k < 8; k++) { int g = order[k]; if (!canBuildOn(p, g)) continue; while (buildStep(p, g, lv, reserve)) {} }
            }
        } else if (t.bpri == 2) {   // focus: finish one set before spending a dollar on the next
            for (int k = 0; k < 8; k++) {
                int g = order[k]; if (!canBuildOn(p, g)) continue;
                while (buildStep(p, g, t.buildMax, reserve)) {}
                bool done = true; for (int i = 0; i < GSZ[g]; i++) if (houses[GSQ[g][i]] < t.buildMax) done = false;
                if (!done) break;
            }
        } else {                    // priority: finish one set first, leftover cash goes to the next
            for (int k = 0; k < 8; k++) { int g = order[k]; if (!canBuildOn(p, g)) continue; while (buildStep(p, g, t.buildMax, reserve)) {} }
        }
    }

    // -------- trading
    static double tvOf(int g) { return g < 8 ? TV[g] : 0.0; }
    double valueOf(int who, int gGet, int gGive, int faceGet, int faceGive) const {
        // value, to `who`, of completing gGet while handing the opponent gGive
        if (S[who]->trade == 2) return tvOf(gGet) - tvOf(gGive);
        return faceGet - faceGive;
    }
    void tryTrades(int a) {
        if (!cfg.trading || S[a]->trade == 0 || P[a].cash < S[a]->tradeCash) return;
        for (int b = 0; b < n; b++) {
            if (b == a || !P[b].alive || S[b]->trade == 0 || P[b].cash < S[b]->tradeCash) continue;
            int cntA[9], cntB[9];
            for (int g = 0; g < 9; g++) { cntA[g] = countOwned(a, g); cntB[g] = countOwned(b, g); }
            int bestGA = -1, bestGB = -1, bestC = 0; double bestGain = -1e18;
            for (int gA = 0; gA < 9; gA++) {
                if (gA == 8 && !S[a]->rrTrade) continue;  // railroads count as a set only for a collector
                if (!(cntA[gA] >= 1 && cntB[gA] >= 1 && cntA[gA] + cntB[gA] == GSZ[gA])) continue;
                for (int gB = 0; gB < 9; gB++) {
                    if (gB == 8 && !S[b]->rrTrade) continue;
                    if (gB == gA || !(cntA[gB] >= 1 && cntB[gB] >= 1 && cntA[gB] + cntB[gB] == GSZ[gB])) continue;
                    if ((S[a]->fav >> GSQ[gB][0] & 1) || (S[b]->fav >> GSQ[gA][0] & 1)) continue;  // fans keep their favourite
                    if (S[a]->trade == 3 && tvOf(gA) < tvOf(gB)) continue;  // picky: never take the weaker set
                    if (S[b]->trade == 3 && tvOf(gB) < tvOf(gA)) continue;
                    int faceFromB = 0, faceFromA = 0;  // list price of what each hands over
                    for (int i = 0; i < GSZ[gA]; i++) if (owner[GSQ[gA][i]] == b) faceFromB += B[GSQ[gA][i]].price;
                    for (int i = 0; i < GSZ[gB]; i++) if (owner[GSQ[gB][i]] == a) faceFromA += B[GSQ[gB][i]].price;
                    // b names its price: cash c from a to b (negative: b pays a)
                    double demand = -valueOf(b, gB, gA, faceFromA, faceFromB);
                    int c = (int)std::ceil(demand);
                    double gain = valueOf(a, gA, gB, faceFromB, faceFromA) - c;
                    if (gain < 0) continue;
                    if (c > 0 && P[a].cash < c) continue;
                    if (c < 0 && P[b].cash < -c) continue;
                    if (gain > bestGain) { bestGain = gain; bestGA = gA; bestGB = gB; bestC = c; }
                }
            }
            if (bestGA < 0) continue;
            for (int i = 0; i < GSZ[bestGA]; i++) { int s = GSQ[bestGA][i]; if (owner[s] == b) owner[s] = a; }
            for (int i = 0; i < GSZ[bestGB]; i++) { int s = GSQ[bestGB][i]; if (owner[s] == a) owner[s] = b; }
            P[a].cash -= bestC; P[b].cash += bestC; st->trades++; lastProgress = turns;
            if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"trade\",\"with\":%d,\"gets\":\"%s\",\"gives\":\"%s\",\"cash\":%d}\n",
                             round, a, b, GROUP_NAME[bestGA], GROUP_NAME[bestGB], bestC);
            updateMono(a); updateMono(b);
        }
        bool stall = stalled();
        if (cfg.buyouts || stall) tryBuyouts(a);
        if (stall) tryConsolidate(a);
    }
    // counted in turns, so the first chance at an endgame deal goes to whoever is next, not always to the first seat
    bool stalled() const { return cfg.endgame > 0 && unownedCount() <= 2 && turns - lastProgress >= (long)cfg.endgame * aliveCount; }
    // endgame: buy one card of a colour that is split between several players, so a set can be put together
    void tryConsolidate(int a) {
        int bestG = -1, bestB = -1, bestS = -1; double bestKey = -1;
        for (int g = 0; g < 8; g++) {
            int mine = countOwned(a, g);
            if (mine == 0 || mine == GSZ[g]) continue;
            int holders = 0, seen[8] = {0}; bool unowned = false;
            for (int i = 0; i < GSZ[g]; i++) { int o = owner[GSQ[g][i]]; if (o < 0) unowned = true; else if (o != a && !seen[o]) { seen[o] = 1; holders++; } }
            if (unowned || holders < 2) continue;  // one holder: an ordinary buyout handles it
            for (int i = 0; i < GSZ[g]; i++) {
                int s = GSQ[g][i], b = owner[s];
                if (b < 0 || b == a || !P[b].alive || S[b]->trade == 0 || S[b]->sellPremium <= 0) continue;
                if (countOwned(b, g) > mine) continue;  // nobody sells to a player with fewer cards of the colour than they have
                double key = (double)mine / GSZ[g] * 10 - countOwned(b, g);
                if (key > bestKey) { bestKey = key; bestG = g; bestB = b; bestS = s; }
            }
        }
        if (bestG < 0) return;
        int price = (int)std::ceil(S[bestB]->sellPremium * B[bestS].price);
        if (S[a]->buyPremium * B[bestS].price < price || P[a].cash - price < S[a]->reserve) return;
        owner[bestS] = a; P[a].cash -= price; P[bestB].cash += price; st->trades++; st->endgameDeals++;
        if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"endgame_buy\",\"from\":%d,\"s\":%d,\"cash\":%d}\n", round, a, bestB, bestS, price);
        updateMono(a);
    }
    // a completes a colour set by buying the missing pieces, all held by one player, for cash
    void tryBuyouts(int a) {
        for (int g = 0; g < 8; g++) {
            int mine = countOwned(a, g); if (mine == 0 || mine == GSZ[g]) continue;
            int b = -1, face = 0; bool ok = true;
            for (int i = 0; i < GSZ[g]; i++) {
                int s = GSQ[g][i], o = owner[s];
                if (o == a) continue;
                if (o < 0 || (b >= 0 && o != b)) { ok = false; break; }
                b = o; face += B[s].price;
            }
            if (!ok || b < 0 || !P[b].alive || S[b]->trade == 0 || S[b]->sellPremium <= 0) continue;
            double ask = S[b]->trade == 2 ? face + std::max(0.0, TV[g]) : S[b]->sellPremium * face;
            double will = S[a]->trade == 2 ? face + std::max(0.0, TV[g]) : S[a]->buyPremium * face;
            int price = (int)std::ceil(ask);
            if (will < ask || P[a].cash - price < S[a]->reserve) continue;
            for (int i = 0; i < GSZ[g]; i++) { int s = GSQ[g][i]; if (owner[s] == b) owner[s] = a; }
            P[a].cash -= price; P[b].cash += price; st->trades++; lastProgress = turns;
            if (!cfg.buyouts) st->endgameDeals++;
            if (log) fprintf(log, "{\"r\":%d,\"p\":%d,\"e\":\"buyout\",\"from\":%d,\"gets\":\"%s\",\"cash\":%d}\n", round, a, b, GROUP_NAME[g], price);
            updateMono(a);
        }
    }

    // -------- turn
    bool wantsOut(int p) const {
        int j = S[p]->jail;
        if (j == 0) return true;
        if (j == 1) return false;
        if (j == 3) {  // stay while an opponent has houses in the stretch you roll into (St. Charles to Illinois)
            for (int s = 11; s <= 24; s++) if (owner[s] >= 0 && owner[s] != p && houses[s] > 0) return false;
            return true;
        }
        if (j == 4) return !anyOppDeveloped(p);
        if (j == 5) return unownedCount() > 4;
        if (j == 6) return P[p].jailTurns >= 1;  // try one roll for doubles, then pay
        return !(anyOppDeveloped(p) || unownedCount() <= 4);
    }
    void turn(int p) {
        turns++;
        tryTrades(p);
        int dbl = 0;
        while (P[p].alive) {
            if (P[p].inJail) {
                if (wantsOut(p)) {
                    if (P[p].goojfCh) { P[p].goojfCh--; chance.bottom(CH_GOOJF); P[p].inJail = false; }
                    else if (P[p].goojfCc) { P[p].goojfCc--; chest.bottom(CC_GOOJF); P[p].inJail = false; }
                    else if (P[p].cash >= 50) { pay(p, POT, 50); P[p].inJail = false; }
                }
                if (P[p].inJail) {
                    int d1 = rng.die(), d2 = rng.die();
                    if (d1 == d2) { P[p].inJail = false; P[p].jailTurns = 0; moveBy(p, d1 + d2, d1 + d2); break; }
                    if (++P[p].jailTurns >= 3) {
                        if (!pay(p, POT, 50)) return;
                        P[p].inJail = false; P[p].jailTurns = 0; moveBy(p, d1 + d2, d1 + d2);
                    }
                    break;
                }
                P[p].jailTurns = 0;
            }
            int d1 = rng.die(), d2 = rng.die();
            if (d1 == d2 && ++dbl == 3) { goJail(p); break; }
            moveBy(p, d1 + d2, d1 + d2);
            if (!P[p].alive || P[p].inJail || d1 != d2) break;
        }
        if (P[p].alive) manage(p);
#ifdef CHECK
        check();
#endif
    }
    void check() const {
        int h = 0, ht = 0;
        for (int s = 0; s < 40; s++) {
            if (houses[s] == 5) ht++; else h += houses[s];
            if (houses[s] && (owner[s] < 0 || !hasSet(owner[s], B[s].group))) { fprintf(stderr, "building without set at %d\n", s); abort(); }
            if (houses[s] && mort[s]) { fprintf(stderr, "building on mortgaged %d\n", s); abort(); }
            if (owner[s] >= 0 && !P[owner[s]].alive) { fprintf(stderr, "dead owner %d\n", s); abort(); }
        }
        if (h + bankHouses != 32 || ht + bankHotels != 12) { fprintf(stderr, "house count %d+%d %d+%d\n", h, bankHouses, ht, bankHotels); abort(); }
        for (int p = 0; p < n; p++) if (P[p].alive && P[p].cash < 0) { fprintf(stderr, "negative cash p%d %d\n", p, P[p].cash); abort(); }
        int goojf = chance.cnt + chest.cnt; for (int p = 0; p < n; p++) goojf += P[p].goojfCh + P[p].goojfCc;
        if (goojf != 32) { fprintf(stderr, "card count %d\n", goojf); abort(); }
    }

    // returns the winning player index, or -1 on timeout
    int play() {
        for (int s = 0; s < 40; s++) { owner[s] = -1; houses[s] = 0; mort[s] = false; }
        for (int p = 0; p < n; p++) { P[p] = Player(); P[p].cash = cfg.startCash; P[p].slot = p; }
        aliveCount = n;
        chance.shuffle(rng); chest.shuffle(rng);
        for (auto& gv : cfg.give) for (int i = 0; i < GSZ[gv.second]; i++) owner[GSQ[gv.second][i]] = gv.first;
        if (cfg.deal) {  // mid-game snapshot: every other property already bought, dealt out at random
            int left[40], nl = 0;
            for (int s = 0; s < 40; s++) if (buyable(s) && owner[s] < 0) left[nl++] = s;
            for (int i = nl - 1; i > 0; i--) std::swap(left[i], left[rng.below(i + 1)]);
            int start = rng.below(n);
            for (int i = 0; i < nl; i++) owner[left[i]] = (start + i) % n;
        }
        for (int p = 0; p < n; p++) updateMono(p);
        // random seating
        int order[8]; for (int i = 0; i < n; i++) order[i] = i;
        for (int i = n - 1; i > 0; i--) std::swap(order[i], order[rng.below(i + 1)]);
        for (int i = 0; i < n; i++) st->seatGames[i]++;
        for (round = 0; round < cfg.maxRounds && aliveCount > 1; round++) {
            for (int k = 0; k < n && aliveCount > 1; k++) {
                int p = order[k];
                if (P[p].alive) turn(p);
            }
        }
        int w = -1;
        if (aliveCount == 1) for (int p = 0; p < n; p++) if (P[p].alive) w = p;
        // stats
        st->games++; st->turnsTotal += turns;
        int nwBest = -1, nwVal = -1;
        for (int p = 0; p < n; p++) if (P[p].alive) { int v = netWorth(p); if (v > nwVal) { nwVal = v; nwBest = p; } }
        for (int p = 0; p < n; p++) {
            st->rent[p] += P[p].rent; st->netAtEnd[p] += P[p].alive ? netWorth(p) : 0;
            if (P[p].monoMask) st->gotMono[p]++;
            if (P[p].rrAll) { st->allRR[p]++; if (p == w) st->allRRWin[p]++; }
            for (int g = 0; g < 8; g++) if (P[p].monoMask >> g & 1) { st->monoBy[p][g]++; if (p == w) st->winMonoBy[p][g]++; }
            if (P[p].firstMono >= 0) st->firstMonoAny[P[p].firstMono]++;
        }
        if (w >= 0) {
            st->wins[w]++; st->nwWins[w]++;
            if (P[w].monoMask) st->winWithMono[w]++;
            if (P[w].firstMono >= 0) st->winnerFirstMono[P[w].firstMono]++;
            for (int k = 0; k < n; k++) if (order[k] == w) st->seatWins[k]++;
            st->roundsSum += round; st->roundsHist[std::min(100, round / 10)]++;
        } else {
            st->timeouts++;
            if (nwBest >= 0) st->nwWins[nwBest]++;
        }
        if (log) fprintf(log, "{\"r\":%d,\"e\":\"end\",\"winner\":%d}\n", round, w);
        return w;
    }
};

// ---------------------------------------------------------------- start-up tables
static void initTables() {
    for (int g = 0; g < 10; g++) GSZ[g] = 0;
    for (int s = 0; s < 40; s++) if (B[s].group >= 0) { int g = B[s].group; GSQ[g][GSZ[g]++] = s; }
}
// landing frequency per turn for one token that leaves jail at once, by simulation
static void estimateFreq(long turns, uint64_t seed) {
    Config cfg; Stats st; st.init(1);
    Strat lone = presetStrat("nothing");
    std::vector<const Strat*> v = {&lone};
    Game g(cfg, seed, v, &st);
    for (int s = 0; s < 40; s++) { g.owner[s] = -1; g.houses[s] = 0; g.mort[s] = false; }
    g.n = 1; g.P[0] = Player(); g.P[0].cash = 1 << 30; g.aliveCount = 1;
    g.chance.shuffle(g.rng); g.chest.shuffle(g.rng);
    for (long t = 0; t < turns; t++) g.turn(0);
    for (int s = 0; s < 40; s++) FREQ[s] = st.landSq[s] / (double)turns;
}
static void initValues(double horizon) {
    double roi[8];
    for (int g = 0; g < 8; g++) {
        double inc = 0, cost = 0;
        for (int i = 0; i < GSZ[g]; i++) { int s = GSQ[g][i]; inc += FREQ[s] * B[s].rent[3]; cost += 3 * B[s].hc; }
        roi[g] = inc / cost;
        TV[g] = horizon * inc - cost;
        BUILD_ORDER[g] = g;
    }
    std::sort(BUILD_ORDER, BUILD_ORDER + 8, [&](int a, int b) { return roi[a] > roi[b]; });
}

// ---------------------------------------------------------------- main
static int groupByName(const std::string& n) { for (int g = 0; g < 10; g++) if (n == GROUP_NAME[g]) return g; return -1; }

int main(int argc, char** argv) {
    initTables();
    Config cfg; long games = 10000; uint64_t seed = 1; int threads = std::max(1u, std::thread::hardware_concurrency());
    std::string strats = "everything|everything|everything|everything", logPath, tvSpec; long freqTurns = 0; double horizon = 25;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--strats") strats = nx();
        else if (a == "--games") games = atol(nx().c_str());
        else if (a == "--seed") seed = strtoull(nx().c_str(), nullptr, 10);
        else if (a == "--threads") threads = atoi(nx().c_str());
        else if (a == "--auctions") cfg.auctions = atoi(nx().c_str());
        else if (a == "--trading") cfg.trading = atoi(nx().c_str());
        else if (a == "--fp") cfg.freeParking = atoi(nx().c_str());
        else if (a == "--go400") cfg.go400 = atoi(nx().c_str());
        else if (a == "--buyouts") cfg.buyouts = atoi(nx().c_str());
        else if (a == "--deal") cfg.deal = atoi(nx().c_str());
        else if (a == "--endgame") cfg.endgame = atoi(nx().c_str());
        else if (a == "--maxrounds") cfg.maxRounds = atoi(nx().c_str());
        else if (a == "--cash") cfg.startCash = atoi(nx().c_str());
        else if (a == "--horizon") horizon = atof(nx().c_str());
        else if (a == "--log") logPath = nx();
        else if (a == "--tv") tvSpec = nx();
        else if (a == "--freq") freqTurns = atol(nx().c_str());
        else if (a == "--give") {
            std::stringstream ss(nx()); std::string t;
            while (std::getline(ss, t, ',')) { auto c = t.find(':'); cfg.give.push_back({atoi(t.substr(0, c).c_str()), groupByName(t.substr(c + 1))}); }
        } else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    estimateFreq(2000000, 12345);
    initValues(horizon);
    if (!tvSpec.empty()) {  // "brown:100,orange:900,..." overrides the shark's set values
        std::stringstream ss(tvSpec); std::string t;
        while (std::getline(ss, t, ',')) { auto c = t.find(':'); int g = groupByName(t.substr(0, c)); if (g >= 0 && g < 8) TV[g] = atof(t.substr(c + 1).c_str()); }
    }
    if (freqTurns) {
        estimateFreq(freqTurns, seed);
        printf("{\"freq\":[");
        for (int s = 0; s < 40; s++) printf("%s%.6f", s ? "," : "", FREQ[s]);
        printf("]}\n");
        return 0;
    }
    std::vector<Strat> SS; { std::stringstream ss(strats); std::string t; while (std::getline(ss, t, '|')) SS.push_back(parseStrat(t)); }
    int np = (int)SS.size();
    std::vector<const Strat*> sp; for (auto& s : SS) sp.push_back(&s);

    Stats total; total.init(np);
    if (!logPath.empty()) {
        Stats st; st.init(np);
        Game g(cfg, seed, sp, &st); g.log = fopen(logPath.c_str(), "w");
        for (int p = 0; p < np; p++) fprintf(g.log, "{\"e\":\"player\",\"p\":%d,\"strat\":\"%s\"}\n", p, SS[p].name.c_str());
        g.play(); fclose(g.log);
    }
    std::atomic<long> next{0}; std::mutex mu;
    auto work = [&]() {
        Stats st; st.init(np);
        for (;;) {
            long i = next.fetch_add(1); if (i >= games) break;
            Game g(cfg, seed * 1000003ull + (uint64_t)i, sp, &st);
            g.play();
        }
        std::lock_guard<std::mutex> lk(mu); total.add(st);
    };
    std::vector<std::thread> th; for (int t = 0; t < threads; t++) th.emplace_back(work);
    for (auto& t : th) t.join();

    // JSON summary
    const Stats& s = total; long fin = s.games - s.timeouts;
    printf("{\"games\":%ld,\"timeouts\":%ld,\"finished\":%ld,\"avg_rounds_finished\":%.2f,\"trades\":%ld,\"turns\":%ld,",
           s.games, s.timeouts, fin, fin ? s.roundsSum / fin : 0.0, s.trades, s.turnsTotal);
    printf("\"endgame_deals\":%ld,\"config\":{\"auctions\":%d,\"trading\":%d,\"fp\":%d,\"go400\":%d,\"buyouts\":%d,\"endgame\":%d,\"maxrounds\":%d,\"cash\":%d},",
           s.endgameDeals, cfg.auctions, cfg.trading, cfg.freeParking, cfg.go400, cfg.buyouts, cfg.endgame, cfg.maxRounds, cfg.startCash);
    printf("\"players\":[");
    for (int i = 0; i < np; i++) {
        printf("%s{\"name\":\"%s\",\"wins\":%ld,\"nw_wins\":%ld,\"got_set\":%ld,\"wins_with_set\":%ld,\"first_out\":%ld,\"busted\":%ld,\"all_rr\":%ld,\"all_rr_win\":%ld,\"avg_rent\":%.1f,\"avg_net_end\":%.1f,\"sets\":{",
               i ? "," : "", SS[i].name.c_str(), s.wins[i], s.nwWins[i], s.gotMono[i], s.winWithMono[i], s.firstOut[i], s.busted[i], s.allRR[i], s.allRRWin[i], s.rent[i] / s.games, s.netAtEnd[i] / s.games);
        for (int g = 0; g < 8; g++) printf("%s\"%s\":[%ld,%ld]", g ? "," : "", GROUP_NAME[g], s.monoBy[i][g], s.winMonoBy[i][g]);
        printf("}}");
    }
    printf("],\"seat_wins\":[");
    for (int i = 0; i < np; i++) printf("%s%ld", i ? "," : "", s.seatWins[i]);
    printf("],\"winner_first_set\":{");
    for (int g = 0; g < 8; g++) printf("%s\"%s\":%ld", g ? "," : "", GROUP_NAME[g], s.winnerFirstMono[g]);
    printf("},\"first_set_any\":{");
    for (int g = 0; g < 8; g++) printf("%s\"%s\":%ld", g ? "," : "", GROUP_NAME[g], s.firstMonoAny[g]);
    printf("},\"rent_by_square\":[");
    for (int q = 0; q < 40; q++) printf("%s%.0f", q ? "," : "", s.rentSq[q]);
    printf("],\"land_by_square\":[");
    for (int q = 0; q < 40; q++) printf("%s%.0f", q ? "," : "", s.landSq[q]);
    printf("],\"rounds_hist_10\":[");
    for (size_t q = 0; q < s.roundsHist.size(); q++) printf("%s%ld", q ? "," : "", s.roundsHist[q]);
    printf("],\"build_order\":[");
    for (int g = 0; g < 8; g++) printf("%s\"%s\"", g ? "," : "", GROUP_NAME[BUILD_ORDER[g]]);
    printf("],\"set_value\":{");
    for (int g = 0; g < 8; g++) printf("%s\"%s\":%.0f", g ? "," : "", GROUP_NAME[g], TV[g]);
    printf("}}\n");
    return 0;
}
