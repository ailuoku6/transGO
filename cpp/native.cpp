#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <set>
#include <unordered_set>
#include <vector>
#include "vendor/katago/game/boardhistory.h"

namespace py = pybind11;

// The public Python Board is a small value type around KataGo's production
// rule core.  The neural net and PUCT search remain local to this project,
// while legality, ko, pass phases, scoring, area, seki and dead-stone status
// all come from the vendored KataGo Board/BoardHistory implementation.
class GoBoard {
public:
    int size;
    int to_play = 1;       // project convention: black=+1, white=-1
    int passes = 0;
    int move_number = 0;
    int last_move = -1;
    int encore_phase = 0;
    double komi;
    std::string rules_name;
    std::vector<int> cells;

private:
    ::Board kboard;
    ::BoardHistory khist;
    std::vector<std::vector<int>> history;

    static ::Board make_kata_board(int n) {
        if (n < 2 || n > ::Board::MAX_LEN)
            throw std::invalid_argument("invalid board size");
        ::Board::initHash();
        return ::Board(n, n);
    }

    static std::string kata_rules_name(const std::string& raw) {
        std::string lower = raw;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        // Keep the historical config spelling, but map it to KataGo's exact
        // positional-superko Chinese preset.
        if (lower == "chinese-area-psk" || lower == "chinese_area_psk" || lower == "area" || lower == "chinese-area")
            return "Chinese-OGS";
        if (lower == "chinese_area")
            return "Chinese";
        if (lower == "territory" || lower == "japanese-territory" || lower == "japanese_territory")
            return "Japanese";
        return raw;
    }

    static ::Rules make_rules(const std::string& raw, double k) {
        if (raw.empty() || !std::isfinite(k))
            throw std::invalid_argument("invalid rules or komi");
        const std::string name = kata_rules_name(raw);
        try {
            return ::Rules::parseRulesWithoutKomi(name, static_cast<float>(k));
        } catch (const std::exception&) {
            // JSON rules may contain a komi field.  Parse it normally and
            // let the explicit Board komi argument remain authoritative.
            ::Rules rules = ::Rules::parseRules(name);
            rules.komi = static_cast<float>(k);
            return rules;
        }
    }

    static ::Player kata_player(int p) {
        if (p == 1) return ::P_BLACK;
        if (p == -1) return ::P_WHITE;
        throw std::invalid_argument("invalid player");
    }

    static int project_color(::Color c) {
        return c == ::C_BLACK ? 1 : (c == ::C_WHITE ? -1 : 0);
    }

    int action_loc(int action) const {
        if (action == size * size) return ::Board::PASS_LOC;
        if (action < 0 || action >= size * size) return ::Board::NULL_LOC;
        return ::Location::getLoc(action % size, action / size, size);
    }

    void sync_cells() {
        cells.assign(size * size, 0);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                const ::Loc loc = ::Location::getLoc(x, y, size);
                cells[y * size + x] = project_color(kboard.colors[loc]);
            }
    }

    void sync_history() {
        sync_cells();
        history.insert(history.begin(), cells);
        if (history.size() > 8) history.pop_back();
        passes = khist.consecutiveEndingPasses;
        encore_phase = khist.encorePhase;
    }

    std::vector<int> area_ownership() const {
        ::Color area[::Board::MAX_ARR_SIZE];
        // Use history for both scoring modes: area rules may also impose
        // seki/all-group tax, and territory depends on encore history.
        khist.getAreaNow(kboard, area);
        std::vector<int> out(size * size, 0);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                const ::Loc loc = ::Location::getLoc(x, y, size);
                out[y * size + x] = project_color(area[loc]);
            }
        return out;
    }

public:
    GoBoard(int n = 9, double k = 7.5, std::string rules = "chinese-area-psk")
        : size(n), komi(k), rules_name(std::move(rules)), kboard(make_kata_board(n)),
          khist(kboard, ::P_BLACK, make_rules(rules_name, k), 0, ::BoardHistoryModes()) {
        if (!std::isfinite(k)) throw std::invalid_argument("invalid komi");
        sync_cells();
        history.push_back(cells);
    }

    bool terminal() const { return khist.isGameFinished; }
    bool no_result() const { return khist.isNoResult; }

    std::string phase() const {
        if (terminal()) return "finished";
        if (khist.rules.scoringRule == ::Rules::SCORING_TERRITORY && encore_phase > 0)
            return encore_phase == 1 ? "encore1" : "encore2";
        return "normal_play";
    }

    bool legal(int action) const {
        if (action < 0 || action > size * size || terminal()) return false;
        const ::Loc loc = action_loc(action);
        return khist.isLegal(kboard, loc, kata_player(to_play));
    }

    std::vector<int> legal_mask() const {
        std::vector<int> out(size * size + 1, 0);
        for (int a = 0; a <= size * size; ++a) out[a] = legal(a) ? 1 : 0;
        return out;
    }

    GoBoard play(int action) const {
        if (!legal(action)) throw std::invalid_argument("illegal move");
        GoBoard out = *this;
        const ::Loc loc = out.action_loc(action);
        out.khist.makeBoardMoveAssumeLegal(out.kboard, loc, kata_player(out.to_play), nullptr);
        out.to_play = -out.to_play;
        out.move_number += 1;
        out.last_move = action;
        out.sync_history();
        return out;
    }

    std::vector<int> ownership() const {
        if (khist.isNoResult) return cells;
        return area_ownership();
    }

    std::vector<int> dead_stones() const {
        if (khist.isNoResult) return std::vector<int>(size * size, 0);
        const auto area = area_ownership();
        std::vector<int> out(size * size, 0);
        for (size_t i = 0; i < cells.size(); ++i)
            if (cells[i] != 0 && area[i] == -cells[i]) out[i] = 1;
        return out;
    }

    std::vector<int> territory() const {
        if (khist.isNoResult) return std::vector<int>(size * size, 0);
        const auto area = area_ownership();
        std::vector<int> out(size * size, 0);
        for (size_t i = 0; i < cells.size(); ++i)
            if (cells[i] == 0) out[i] = area[i];
        return out;
    }

    std::vector<int> seki() const {
        if (khist.isNoResult) return std::vector<int>(size * size, 0);
        ::Color full[::Board::MAX_ARR_SIZE];
        ::Color independent[::Board::MAX_ARR_SIZE];
        int independent_region_count = 0;
        const bool suicide = khist.suicideLegalForPassAlive();
        kboard.calculateArea(full, true, true, true, suicide);
        kboard.calculateIndependentLifeArea(
            independent, independent_region_count, false, false,
            khist.modes.excludeTerritoryAdjacentToAtari, suicide);
        std::vector<int> out(size * size, 0);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                const ::Loc loc = ::Location::getLoc(x, y, size);
                // KataGo calls a point seki when it is in the full area but
                // not in an independent-life area after dame/atari stripping.
                out[y * size + x] = (full[loc] != ::C_EMPTY && independent[loc] == ::C_EMPTY) ? 1 : 0;
            }
        return out;
    }

    double score() const {
        if (khist.isNoResult) return 0.0;
        // KataGo stores finalWhiteMinusBlackScore.  This project exposes
        // black-minus-white, matching the historical Python API.
        ::BoardHistory scored = khist;
        scored.endAndScoreGameNow(kboard);
        if (scored.isNoResult) return 0.0;
        return -static_cast<double>(scored.finalWhiteMinusBlackScore);
    }

    double result() const {
        if (khist.isNoResult) return 0.0;
        const double s = score();
        return s > 0.0 ? 1.0 : (s < 0.0 ? -1.0 : 0.0);
    }

    uint64_t hash() const { return kboard.pos_hash.hash0 ^ kboard.pos_hash.hash1; }

    py::dict formal_result() const {
        py::dict out;
        out["score_black"] = score();
        out["winner"] = result();
        out["ownership"] = ownership();
        out["dead_stones"] = dead_stones();
        out["territory"] = territory();
        out["seki"] = seki();
        out["phase"] = phase();
        out["rules"] = rules_name;
        out["encore_phase"] = encore_phase;
        out["terminal"] = terminal();
        out["no_result"] = no_result();
        out["adjudication_engine"] = "katago-boardhistory";
        return out;
    }

    static GoBoard from_position(int n, double k, const std::vector<int>& position,
                                 int next_player, std::string rules) {
        if (position.size() != static_cast<size_t>(n * n) ||
            (next_player != 1 && next_player != -1))
            throw std::invalid_argument("invalid analysis position");
        GoBoard board(n, k, std::move(rules));
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const int color = position[y * n + x];
                if (color < -1 || color > 1) throw std::invalid_argument("invalid stone color");
                const ::Loc loc = ::Location::getLoc(x, y, n);
                board.kboard.colors[loc] = color == 1 ? ::C_BLACK : (color == -1 ? ::C_WHITE : ::C_EMPTY);
            }
        board.kboard.regenChainsFromColors();
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const ::Loc loc = ::Location::getLoc(x, y, n);
                if (board.kboard.colors[loc] != ::C_EMPTY && board.kboard.getNumLiberties(loc) == 0)
                    throw std::invalid_argument("analysis position contains a group with no liberties");
            }
        board.khist = ::BoardHistory(board.kboard, kata_player(next_player),
                                     make_rules(board.rules_name, k), 0, ::BoardHistoryModes());
        board.to_play = next_player;
        board.move_number = 0;
        board.last_move = -1;
        board.sync_cells();
        board.history.clear();
        board.history.push_back(board.cells);
        return board;
    }

    py::array_t<float> features() const {
        py::array_t<float> a({21, size, size});
        auto f = a.mutable_unchecked<3>();
        const auto legal_moves = legal_mask();
        for (int p = 0; p < size * size; ++p) {
            for (int t = 0; t < 8; ++t) {
                const int c = t < static_cast<int>(history.size()) ? history[t][p] : 0;
                f(2 * t, p / size, p % size) = c == to_play;
                f(2 * t + 1, p / size, p % size) = c == -to_play;
            }
            f(16, p / size, p % size) = legal_moves[p];
            f(17, p / size, p % size) = p == last_move;
            f(18, p / size, p % size) = to_play == 1;
            f(19, p / size, p % size) = static_cast<float>(-komi * to_play / (size * size));
            f(20, p / size, p % size) = static_cast<float>(move_number) / (size * size);
        }
        return a;
    }
};

struct Node {
    GoBoard board;
    int visits=0, virtual_visits=0;
    double value_sum=0;
    bool expanded=false, pending=false;
    std::vector<double> priors;
    std::vector<int> legal;
    std::vector<std::unique_ptr<Node>> children;
    explicit Node(const GoBoard& b, int min_moves=0):board(b),priors(b.size*b.size+1,0),legal(b.legal_mask()),children(b.size*b.size+1) {
        // A self-play exploration constraint, not a change to Go's legal moves.
        // Always retain pass when no board move is legal.
        if(b.move_number<min_moves && std::any_of(legal.begin(),legal.end()-1,[](int v){return v!=0;}))
            legal.back()=0;
    }
};

class Search {
    double c_puct, virtual_loss;
    std::mt19937 rng;
    using Path=std::vector<Node*>;
    void backup(const Path& path,double value) {
        for(auto it=path.rbegin();it!=path.rend();++it) {
            (*it)->visits++; (*it)->value_sum+=value; value=-value;
        }
    }
    void expand(Node* n,const float* p) {
        const auto& legal=n->legal;double sum=0;
        for(size_t a=0;a<legal.size();++a) {
            if(!std::isfinite(p[a]) || p[a]<0) throw std::invalid_argument("invalid network policy");
            n->priors[a]=legal[a] ? p[a] : 0;sum+=n->priors[a];
        }
        if(sum<=0) {
            for(size_t a=0;a<legal.size();++a) n->priors[a]=legal[a];
            sum=std::accumulate(n->priors.begin(),n->priors.end(),0.0);
        }
        for(double& p0:n->priors) p0/=sum;
        n->expanded=true;
    }
    Path select(Node& root, int min_moves) {
        Path path{&root}; Node* n=&root;
        while(n->expanded && !n->board.terminal()) {
            int best=-1;double best_score=-1e100;
            double total=std::sqrt(static_cast<double>(n->visits+n->virtual_visits+1));
            const auto& mask=n->legal;
            for(size_t a=0;a<n->priors.size();++a) if(mask[a]) {
                Node* ch=n->children[a].get();
                if(ch && ch->pending) continue;
                int count=ch ? ch->visits+ch->virtual_visits : 0;
                // Child W is in child's player perspective, so negate for parent.
                double q=count ? -(ch->value_sum+virtual_loss*ch->virtual_visits)/count : 0;
                double u=c_puct*n->priors[a]*total/(1+count);
                if(q+u>best_score) {best_score=q+u;best=static_cast<int>(a);}
            }
            if(best<0) return {};
            if(!n->children[best]) n->children[best]=std::make_unique<Node>(n->board.play(best),min_moves);
            n=n->children[best].get();path.push_back(n);
        }
        if(n->pending) return {};
        return path;
    }
public:
    Search(double cp=1.5,double vl=1.0,unsigned seed=0):c_puct(cp),virtual_loss(vl),rng(seed) {
        if(!std::isfinite(cp) || !std::isfinite(vl) || cp<=0 || vl<0) throw std::invalid_argument("invalid search parameters");
    }
    py::dict run(const GoBoard& board,py::function evaluator,int simulations=200,int batch_size=16,
                 double alpha=0.15,double epsilon=0.0,int min_moves=0,double score_utility=0.0) {
        if(board.terminal() || simulations<1 || batch_size<1 || min_moves<0 || !std::isfinite(alpha) || !std::isfinite(epsilon) ||
           !std::isfinite(score_utility) || alpha<=0 || epsilon<0 || epsilon>1 || score_utility<0)
            throw std::invalid_argument("invalid search request");
        Node root(board,min_moves);
        auto evaluate=[&](const std::vector<Path>& paths) {
            py::list boards;
            for(const auto& path:paths) boards.append(py::cast(path.back()->board));
            py::tuple out=evaluator(boards).cast<py::tuple>();
            if(out.size()!=2 && out.size()!=3) throw std::invalid_argument("evaluator must return (policies, values[, scores])");
            auto p=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(out[0]);
            auto v=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(out[1]);
            py::array_t<float,py::array::c_style|py::array::forcecast> s;
            if(out.size()==3) s=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(out[2]);
            int actions=board.size*board.size+1;
            if(!p || !v || p.ndim()!=2 || p.shape(0)!=static_cast<int>(paths.size()) || p.shape(1)!=actions ||
               v.ndim()!=1 || v.shape(0)!=static_cast<int>(paths.size()) ||
               (out.size()==3 && (!s || s.ndim()!=1 || s.shape(0)!=static_cast<int>(paths.size()))))
                throw std::invalid_argument("invalid evaluator shapes");
            for(size_t i=0;i<paths.size();++i) {
                if(!std::isfinite(v.data()[i]) || std::abs(v.data()[i])>1.001)
                    throw std::invalid_argument("value outside [-1,1]");
                double utility=v.data()[i];
                if(score_utility>0) {
                    if(out.size()!=3 || !std::isfinite(s.data()[i]))
                        throw std::invalid_argument("score utility requires finite scores");
                    utility += score_utility*std::tanh(static_cast<double>(s.data()[i]));
                    utility=std::clamp(utility,-1.0,1.0);
                }
                Node* leaf=paths[i].back();expand(leaf,p.data()+i*actions);leaf->pending=false;
                for(Node* n:paths[i]) n->virtual_visits--;
                backup(paths[i],utility);
            }
        };
        // Root expansion is initialization and excluded from the simulation budget.
        root.virtual_visits++; evaluate({Path{&root}});
        root.visits=0;root.value_sum=0;
        if(epsilon>0) {
            std::gamma_distribution<double> gamma(alpha,1.0);
            const auto& mask=root.legal;std::vector<double> noise(mask.size(),0);double total=0;
            for(size_t a=0;a<mask.size();++a) if(mask[a]) {noise[a]=gamma(rng);total+=noise[a];}
            if(total<=0 || !std::isfinite(total)) throw std::invalid_argument("Dirichlet noise underflow: increase alpha");
            for(size_t a=0;a<mask.size();++a) root.priors[a]=(1-epsilon)*root.priors[a]+epsilon*noise[a]/total;
        }
        int done=0;
        while(done<simulations) {
            std::vector<Path> paths;
            int target=std::min(batch_size,simulations-done);
            for(int i=0;i<target;++i) {
                Path path=select(root,min_moves);if(path.empty()) break;
                Node* leaf=path.back();
                if(leaf->board.terminal()) {backup(path,leaf->board.result()*leaf->board.to_play);done++;}
                else {
                    leaf->pending=true;
                    for(Node* n:path) n->virtual_visits++;
                    paths.push_back(std::move(path));
                }
            }
            if(!paths.empty()) {evaluate(paths);done+=static_cast<int>(paths.size());}
            else if(done<simulations && root.visits!=done) throw std::runtime_error("search stalled");
        }
        std::vector<int> visits(root.children.size(),0);
        std::vector<double> policy(root.children.size(),0);
        for(size_t a=0;a<visits.size();++a) if(root.children[a]) {
            visits[a]=root.children[a]->visits;policy[a]=static_cast<double>(visits[a])/simulations;
        }
        py::dict result;result["visits"]=visits;result["policy"]=policy;
        result["value"]=root.value_sum/root.visits;result["simulations"]=done;
        return result;
    }
};

PYBIND11_MODULE(_native,m) {
    m.doc()="C++17 KataGo BoardHistory rules and batched PUCT search";
    py::class_<GoBoard>(m,"Board")
        .def(py::init<int,double,std::string>(),py::arg("size")=9,py::arg("komi")=7.5,
             py::arg("rules")="chinese-area-psk")
        .def_readonly("size",&GoBoard::size).def_readonly("komi",&GoBoard::komi)
        .def_readonly("rules",&GoBoard::rules_name)
        .def_readonly("to_play",&GoBoard::to_play).def_readonly("passes",&GoBoard::passes)
        .def_readonly("move_number",&GoBoard::move_number).def_readonly("last_move",&GoBoard::last_move)
        .def_readonly("encore_phase",&GoBoard::encore_phase)
        .def_property_readonly("cells",[](const GoBoard& b){return b.cells;})
        .def_static("from_position",&GoBoard::from_position,py::arg("size"),py::arg("komi"),
                    py::arg("cells"),py::arg("to_play")=1,py::arg("rules")="chinese-area-psk")
        .def("play",&GoBoard::play).def("legal",&GoBoard::legal).def("legal_mask",&GoBoard::legal_mask)
        .def("terminal",&GoBoard::terminal).def("score",&GoBoard::score).def("result",&GoBoard::result)
        .def("no_result",&GoBoard::no_result)
        .def("formal_result",&GoBoard::formal_result)
        .def("phase",&GoBoard::phase).def("ownership",&GoBoard::ownership)
        .def("dead_stones",&GoBoard::dead_stones).def("territory",&GoBoard::territory)
        .def("seki",&GoBoard::seki)
        .def("features",&GoBoard::features).def("hash",&GoBoard::hash);
    py::class_<Search>(m,"Search")
        .def(py::init<double,double,unsigned>(),py::arg("c_puct")=1.5,py::arg("virtual_loss")=1.0,py::arg("seed")=0)
        .def("run",&Search::run,py::arg("board"),py::arg("evaluator"),py::arg("simulations")=200,
             py::arg("batch_size")=16,py::arg("alpha")=0.15,py::arg("epsilon")=0.0,
             py::arg("min_moves")=0,py::arg("score_utility")=0.0);
}
