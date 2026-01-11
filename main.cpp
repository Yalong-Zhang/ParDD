#include <bits/stdc++.h>
#include <chrono>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <omp.h>
#include <unordered_set>
#include <latch>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif
using namespace std;
const int INF = 2000000000;
typedef long long ll;

inline int read_number(FILE* in) { int x = 0; char ch = 0; while (ch < '0' || ch > '9') ch = fgetc(in); while (ch >= '0' && ch <= '9') { x = x * 10 + (ch - '0'); ch = fgetc(in); } return x; }
inline void check(bool flag, const char* message) {
	if (!flag) {
		printf("!!!!! CHECK ERROR !!!!!\n Error message: %s\n", message);
		assert(0);
	}
}
struct Timer { chrono::high_resolution_clock::time_point start_time, end_time; void start() { start_time = chrono::high_resolution_clock::now(); } void end() { end_time = chrono::high_resolution_clock::now(); } double time() { return chrono::duration<double>(end_time - start_time).count(); } };

thread_local int g_thread_index = -1;

class ThreadPool_Priority {
public:
	ThreadPool_Priority() = default;
	~ThreadPool_Priority() { shutdown(); }
	ThreadPool_Priority(const ThreadPool_Priority&) = delete;
	ThreadPool_Priority& operator=(const ThreadPool_Priority&) = delete;

	void create_threads(size_t nThreads) {
		stopping.store(false, memory_order_relaxed);
		workers.reserve(nThreads);
		for (size_t i = 0; i < nThreads; ++i) {
			workers.emplace_back([this, i] {
				g_thread_index = static_cast<int>(i);
				worker_loop();
				});
		}
	}
	template <class F>
	void spawn(ll weight, F&& func) {
		pending.fetch_add(1, memory_order_relaxed);
		{
			lock_guard<mutex> lg(mtx);
			q.push(Task{ weight, next_seq++, function<void()>(forward<F>(func)) });
		}
		cv.notify_one();
	}
	void spawn(function<void()> func) { spawn(0, move(func)); }
	void wait_for_empty_queue() {
		unique_lock<mutex> lk(mtx);
		cv_empty.wait(lk, [this] { return q.empty(); });
	}
	void wait_for_all() {
		unique_lock<mutex> lk(mtx);
		cv_done.wait(lk, [this] { return q.empty() && pending.load(memory_order_acquire) == 0; });
	}
	void shutdown() {
		if (workers.empty()) return;
		{
			unique_lock<mutex> lk(mtx);
			stopping.store(true, memory_order_relaxed);
		}
		cv.notify_all();
		for (auto& t : workers) { if (t.joinable()) t.join(); }
		workers.clear();
		stopping.store(false, memory_order_relaxed);
		g_thread_index = -1;
	}
	size_t size() const noexcept { return workers.size(); }
private:
	struct Task { ll weight; uint64_t seq; function<void()> func; };
	struct Cmp { bool operator()(Task const& a, Task const& b) const { if (a.weight != b.weight) return a.weight < b.weight;	return a.seq > b.seq; } };
	void worker_loop() {
		for (;;) {
			function<void()> task;
			{
				unique_lock<mutex> lk(mtx);
				cv.wait(lk, [this] {
					return stopping.load(memory_order_relaxed) || !q.empty();
					});
				if (stopping.load(memory_order_relaxed) && q.empty())
					break;

				task = move(q.top().func);
				q.pop();
				if (q.empty()) cv_empty.notify_all();
			}
			task();
			if (pending.fetch_sub(1, memory_order_acq_rel) == 1) {
				unique_lock<mutex> lk(mtx);
				cv_done.notify_all();
			}
		}
	}
	vector<thread> workers;
	priority_queue<Task, vector<Task>, Cmp> q;
	mutex mtx;
	condition_variable cv, cv_empty, cv_done;
	atomic<uint64_t> pending{ 0 };
	atomic<bool> stopping{ false };
	uint64_t next_seq{ 0 };
};
int NUM_THREADS;
static ThreadPool_Priority pool;
static mutex g_print_mutex;
vector<mutex> idn_locks(16384);

template <class T> struct Set { T* element; bool* in; int size = -1; Set() {} Set(int sz) { size = 0; element = (T*)malloc(sz * sizeof(T)); in = (bool*)malloc(sz * sizeof(bool)); memset(in, 0, sz * sizeof(bool)); } ~Set() { free(element), free(in); } void alloc(int sz) { size = 0; element = (T*)malloc(sz * sizeof(T)); in = (bool*)malloc(sz * sizeof(bool)); memset(in, 0, sz * sizeof(bool)); } void insert(T x) { element[size++] = x; in[x] = true; } void clear() { for (int i = 0; i < size; i++) in[element[i]] = false; size = 0; } };
template <class T> struct Map { int* element; bool* in; int size = -1; T* value; Map() {} Map(int sz) { size = 0; element = (int*)malloc(sz * sizeof(int)); in = (bool*)malloc(sz * sizeof(bool)); memset(in, 0, sz * sizeof(bool)); value = (T*)malloc(sz * sizeof(T)); memset(value, 0, sz * sizeof(T)); } ~Map() { free(element), free(in), free(value); } void alloc(int sz) { size = 0; element = (int*)malloc(sz * sizeof(int)); in = (bool*)malloc(sz * sizeof(bool)); memset(in, 0, sz * sizeof(bool)); value = (T*)malloc(sz * sizeof(T)); memset(value, 0, sz * sizeof(T)); } void freememory() { free(element), free(in), free(value); } void clear() { for (int i = 0; i < size; i++) in[element[i]] = false, value[element[i]] = 0; size = 0; } T& operator[](int x) { if (!in[x]) element[size++] = x, in[x] = true; return value[x]; } };
template <class T> struct Queue { T* element; int head, tail; Queue() {} Queue(int size) { element = (T*)malloc(size * sizeof(T)); head = tail = 0; } ~Queue() { free(element); } void alloc(int sz) { head = tail = 0; element = (T*)malloc(sz * sizeof(T)); } bool empty() { return head == tail; } void clear() { head = tail = 0; } int pop() { return element[head++]; } void push(T x) { element[tail++] = x; } };

enum Algorithm { ENU, DC, COREDC, BINARYDC, LOADDC, CLDC }; Algorithm algorithm_used;

struct Edge { int u, v; }; struct DirectedEdge { int u, v, to; };
struct FlowNetwork;
struct FlowNetwork_Div;
struct Graph {
	int V, E;
	Edge* e; int* undeg; int** adj;
	void read_graph_from_dataset(char* dataset_address);

	int* idn, pseudoarboricity;

	void get_idn_enu();

	int* sorted, * position;
	void change_sorted(int position_l, int position_u, int k_m);
	void get_idn_dc(); void divide_and_conquer(int k_l, int k_u);

	int* core, degeneracy;
	void get_core();
	void get_D_from_core(int i, Set<int>& nodes);
	void get_idn_coredc();

	Set<int>* D; int* layer_edges; bool* have_computed;
	void get_idn_binarydc(); void divide_and_conquer_binarydc(int k_l, int k_u);

	void get_idn_loaddc();
	void get_idn_cldc();

	void divide_and_conquer_no_construct(int k_l, int k_u, bool* orientation, int* indeg, vector<FlowNetwork_Div>& network);

	void output_idn();
};
struct FlowNetwork {
	int V, E;
	DirectedEdge* e; int* undeg, * indeg; int** adj;
	Map<int> new_to_ori, ori_to_new;

	void construct_from_set(Graph& G, Set<int>& element, bool additional_indegree);
	void construct_from_set_core(Graph& G, Set<int>& element, int additional_indegree_core_value);
	void construct_from_idn(Graph& G, int k_l, int k_u, bool additional_indegree);

	int pivot; Queue<int> Q; Map<int> parent, dist, cur; Set<int> vis;
	bool DinicBFS(); bool DinicDFS(int x);
	bool in_S(int x) { return indeg[x] > pivot; } bool in_T(int x) { return indeg[x] < pivot; }
	void get_D(int k);

	bool memory_have_released = false;
	~FlowNetwork() { free_memory(); }
	void free_memory() {
		if (!memory_have_released) {
			free(e), free(undeg), free(indeg);
			for (int x = 0; x < V; x++) free(adj[x]); free(adj);
			memory_have_released = true;
		}
	}
};
struct FlowNetwork_Enu {
	Graph& G; FlowNetwork_Enu(Graph& g) : G(g) {}
	int* orientation;
	int* indeg;
	void init_orientation();

	int pivot; Queue<int> Q;
	int* parent, * cur;
	int16_t* epoch, * dist; int now_epoch;
	bool DinicBFS(); bool DinicDFS(int x);
	bool in_S(int x) { return indeg[x] > pivot; } bool in_T(int x) { return indeg[x] < pivot; }
	void get_D(int k);

	bool memory_have_released = false;
	~FlowNetwork_Enu() { free_memory(); }
	void free_memory() {
		if (!memory_have_released) {
			free(orientation), free(indeg), free(epoch), free(parent), free(dist), free(cur);
			memory_have_released = true;
		}
	}
};
struct FlowNetwork_Div {
	Graph& G; FlowNetwork_Div(Graph& g, bool* orientation_, int* indeg_) : G(g), orientation(orientation_), indeg(indeg_) {}
	bool* orientation;
	int* indeg;

	int pivot, k_l, k_u, now_D_size; Queue<int> Q;
	Map<int> parent, dist, cur; Set<int> vis;
	bool DinicBFS(); bool DinicDFS(int x);
	bool in_S(int x) { return indeg[x] > pivot; } bool in_T(int x) { return indeg[x] < pivot; }
	void get_D(int k, int k_l_, int k_u_);
	void init();

	bool memory_have_released = false;
	~FlowNetwork_Div() { free_memory(); }
	void free_memory() {
		if (!memory_have_released) {
			memory_have_released = true;
		}
	}
};

void Graph::read_graph_from_dataset(char* dataset_address) {
	FILE* in = fopen(dataset_address, "r");
	check(in != NULL, "Can not open file dataset_address");

	E = read_number(in), V = read_number(in); V++;

	e = (Edge*)malloc(E * sizeof(Edge));
	undeg = (int*)malloc(V * sizeof(int));
	memset(undeg, 0, V * sizeof(int));
	adj = (int**)malloc(V * sizeof(int*));
	idn = (int*)malloc(V * sizeof(int));

	if (algorithm_used == DC || algorithm_used == COREDC || algorithm_used == BINARYDC || algorithm_used == LOADDC || algorithm_used == CLDC) {
		sorted = (int*)malloc(V * sizeof(int)), position = (int*)malloc(V * sizeof(int));
	}

	for (int i = 0; i < E; i++) {
		e[i].u = read_number(in), e[i].v = read_number(in);
		if (i % 2 == 0) swap(e[i].u, e[i].v);
		undeg[e[i].u]++, undeg[e[i].v]++;
	}
	for (int x = 0; x < V; x++) adj[x] = (int*)malloc(undeg[x] * sizeof(int));

	if (algorithm_used == DC || algorithm_used == COREDC || algorithm_used == BINARYDC || algorithm_used == CLDC) {
		core = (int*)malloc(V * sizeof(int));
	}

	memset(undeg, 0, V * sizeof(int));
	for (int i = 0; i < E; i++) {
		int u = e[i].u, v = e[i].v;
		adj[u][undeg[u]++] = adj[v][undeg[v]++] = i;
	}
}
void Graph::get_idn_enu() {
	memset(idn, 0, V * sizeof(int));

	vector<FlowNetwork_Enu> network;
	network.reserve(NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++) network.emplace_back(*this);
	for (int i = 0; i < NUM_THREADS; i++) pool.spawn([&network, i] { network[i].init_orientation(); });
	pool.wait_for_all();

	pseudoarboricity = -1;
	int k = 1;
	while (true) {
		pool.wait_for_empty_queue();
		if (pseudoarboricity != -1) break;
		pool.spawn([&network, k] {
			network[g_thread_index].get_D(k);
			});
		k++;
	}
	pool.wait_for_all();

}
void Graph::get_idn_dc() {
	memset(idn, 0, V * sizeof(int));

	get_core();
	degeneracy = 0; for (int x = 0; x < V; x++) degeneracy = max(degeneracy, core[x]);

	int* indeg = (int*)malloc(V * sizeof(int)); memset(indeg, 0, V * sizeof(int));
	bool* orientation = (bool*)malloc(E * sizeof(bool));

	{
		int edge_chunk = E / NUM_THREADS;
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			pool.spawn(0, [this, l, r, orientation, indeg] {
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (indeg[u] > indeg[v]) orientation[i] = true, atomic_ref<int>(indeg[v]).fetch_add(1, memory_order_relaxed);
					else orientation[i] = false, atomic_ref<int>(indeg[u]).fetch_add(1, memory_order_relaxed);
				}
				});
		}
		pool.wait_for_all();
	}

	vector<FlowNetwork_Div> network;
	network.reserve(NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++) network.emplace_back(*this, orientation, indeg);
	for (int i = 0; i < NUM_THREADS; i++) pool.spawn(0, [&network, i] { network[i].init(); });
	pool.wait_for_all();

	int appro_pseudoarboricity = degeneracy;
	for (int x = 0; x < V; x++) sorted[x] = x;
	position[0] = 0, position[appro_pseudoarboricity + 1] = V;

	pool.spawn(0, [this, appro_pseudoarboricity, orientation, indeg, &network] { divide_and_conquer_no_construct(0, appro_pseudoarboricity + 1, orientation, indeg, network); });
	pool.wait_for_all();
	free(indeg), free(orientation);
}
void Graph::get_idn_coredc() {
	memset(idn, 0, V * sizeof(int));
	get_core();

	degeneracy = 0; for (int x = 0; x < V; x++) degeneracy = max(degeneracy, core[x]);
	int flownetwork_cnt = 0;
	for (int p = 2; p <= degeneracy; p *= 2) flownetwork_cnt++;
	Set<int>* nodes = (Set<int>*)malloc((flownetwork_cnt + 1) * sizeof(Set<int>)); for (int i = 0; i <= flownetwork_cnt; i++) nodes[i].alloc(V);
	for (int x = 0; x < V; x++) {
		if (core[x] == 0 || core[x] == 1) idn[x] = core[x], nodes[0].insert(x);
		else {
			int y = core[x], k = 0;
			if (y >= (1 << 16)) y >>= 16, k += 16; if (y >= (1 << 8)) y >>= 8, k += 8; if (y >= (1 << 4)) y >>= 4, k += 4; if (y >= (1 << 2)) y >>= 2, k += 2; if (y >= (1 << 1)) k += 1;
			nodes[k].insert(x);	idn[x] = 1 << (k - 1);
		}
	}
	for (int i = 1; i <= flownetwork_cnt; i++) pool.spawn([this, i, nodes] {get_D_from_core(i, nodes[i]); });
	pool.wait_for_all();
	int* indeg = (int*)malloc(V * sizeof(int)); memset(indeg, 0, V * sizeof(int));
	bool* orientation = (bool*)malloc(E * sizeof(bool));
	for (int k = 0; k <= flownetwork_cnt; k++) {
		for (int i = 0; i < nodes[k].size; i++) {
			int x = nodes[k].element[i];
			for (int j = 0; j < undeg[x]; j++) {
				Edge& ne = e[adj[x][j]];
				int y = ne.u == x ? ne.v : ne.u;
				if (ne.u == y) continue;
				if (idn[ne.u] > idn[ne.v]) orientation[adj[x][j]] = true, atomic_ref<int>(indeg[ne.v]).fetch_add(1, memory_order_relaxed);
				else if (idn[ne.v] > idn[ne.u]) orientation[adj[x][j]] = false, atomic_ref<int>(indeg[ne.u]).fetch_add(1, memory_order_relaxed);
				else orientation[adj[x][j]] = true, atomic_ref<int>(indeg[ne.v]).fetch_add(1, memory_order_relaxed);
			}
		}
	}
	for (int i = 0; i <= flownetwork_cnt; i++) nodes[i].~Set<int>(); free(nodes);
	for (int x = 0; x < V; x++) sorted[x] = x;
	auto cmp = [this](int x, int y) { return idn[x] < idn[y]; };
	sort(sorted, sorted + V, cmp);
	int now_idn = -1;
	for (int i = 0; i < V; i++)	while (idn[sorted[i]] > now_idn) position[++now_idn] = i;

	vector<FlowNetwork_Div> network;
	network.reserve(NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++) network.emplace_back(*this, orientation, indeg);
	for (int i = 0; i < NUM_THREADS; i++) pool.spawn([&network, i] { network[i].init(); });
	pool.wait_for_all();

	for (int i = 1; i <= flownetwork_cnt; i++) {
		if ((1 << i) > now_idn) position[1 << i] = V;
		if ((1 << (i + 1)) > now_idn) position[1 << (i + 1)] = V;
		pool.spawn((position[1 << (i + 1)] - position[1 << i]) * ll(1 << i), [this, i, orientation, indeg, &network] { divide_and_conquer_no_construct(1 << i, 1 << (i + 1), orientation, indeg, network); });
	}
	pool.wait_for_all();
	free(indeg), free(orientation);
}
void Graph::get_idn_binarydc() {
	memset(idn, 0, V * sizeof(int));
	get_core();
	degeneracy = 0; for (int x = 0; x < V; x++) degeneracy = max(degeneracy, core[x]);
	D = (Set<int>*)malloc((degeneracy + 2) * sizeof(Set<int>));
	layer_edges = (int*)malloc((degeneracy + 2) * sizeof(int));
	have_computed = (bool*)malloc((degeneracy + 2) * sizeof(bool)); memset(have_computed, 0, (degeneracy + 2) * sizeof(bool));
	for (int k = 0; k <= degeneracy + 1; k++) D[k].alloc(V);
	for (int x = 0; x < V; x++) sorted[x] = x, D[0].insert(x);
	layer_edges[0] = E, layer_edges[degeneracy + 1] = 0; have_computed[0] = have_computed[degeneracy + 1] = true;
	pool.spawn([this] { divide_and_conquer_binarydc(0, degeneracy + 1); });
	pool.wait_for_all();
	for (int k = 0; k <= degeneracy + 1; ++k) D[k].~Set<int>(); free(D);
	free(layer_edges); free(have_computed);
}
static void parallel_expand_reverse_frontier(ThreadPool_Priority& pool, const Graph& G, const int* indeg, const bool* orientation, const vector<int>& frontier, vector<atomic<uint8_t>>& visited, vector<int>& out_new_nodes, int& now_low_indeg) {
	out_new_nodes.clear();
	const size_t T = max<size_t>(1, pool.size());
	const size_t n = frontier.size();
	if (n == 0) return;

	const size_t chunk = (n + T - 1) / T;

	vector<vector<int>> next_local(T);
	vector<int>         local_min(T, INT_MAX);

	auto work = [&](size_t L, size_t R, int tid) {
		int myMin = INT_MAX;
		auto& out = next_local[tid];
		out.reserve((R - L) * 2);
		for (size_t i = L; i < R; ++i) {
			int x = frontier[i];
			for (int j = 0; j < G.undeg[x]; ++j) {
				int id = G.adj[x][j];
				const Edge& ne = G.e[id];
				if (orientation[id] && x == ne.u) continue;
				if (!orientation[id] && x == ne.v) continue;
				int from = (x == ne.u ? ne.v : ne.u);
				uint8_t expect = 0;
				if (visited[from].compare_exchange_strong(expect, 1, memory_order_acq_rel)) {
					out.push_back(from);
					myMin = min(myMin, indeg[from]);
				}
			}
		}
		local_min[tid] = myMin;
	};

	if (T == 1) {
		work(0, n, 0);
	}
	else {
		for (size_t t = 0; t < T; ++t) {
			const size_t L = t * chunk;
			if (L >= n) break;
			const size_t R = min(n, (t + 1) * chunk);
			pool.spawn([&, L, R] {
				int tid = (g_thread_index >= 0 ? g_thread_index : (int)t);
				work(L, R, tid);
				});
		}
		pool.wait_for_all();
	}

	int min_new = now_low_indeg;
	for (int m : local_min) if (m != INT_MAX) min_new = min(min_new, m);
	now_low_indeg = min_new;

	size_t total = 0;
	for (auto& v : next_local) total += v.size();
	out_new_nodes.reserve(total);
	for (auto& v : next_local) {
		if (!v.empty()) {
			out_new_nodes.insert(out_new_nodes.end(), v.begin(), v.end());
		}
	}
}
void get_D_from_orientation_parallel(Graph& G, const int* indeg, const bool* orientation, ThreadPool_Priority& pool, vector<int>& find_D_number, int max_indeg) {
	const int V = G.V;

	vector<vector<int>> indeg_nodes(max_indeg + 1);
	for (int x = 0; x < V; ++x) indeg_nodes[indeg[x]].push_back(x);

	vector<atomic<uint8_t>> visited(V);
	for (int i = 0; i < V; ++i) visited[i].store(0, memory_order_relaxed);
	vector<int> vis_order;  vis_order.reserve(V);
	vector<int> frontier;   frontier.reserve(V / 8);
	vector<int> new_nodes;  new_nodes.reserve(V / 8);

	auto push_bucket_as_seeds = [&](int r) {
		if (r < 0 || r > max_indeg) return;
		for (int x : indeg_nodes[r]) {
			uint8_t expect = 0;
			if (visited[x].compare_exchange_strong(expect, 1, memory_order_acq_rel)) {
				frontier.push_back(x);
				vis_order.push_back(x);
			}
		}
	};

	int now_low_indeg = max_indeg, now_tail = 0;
	find_D_number.clear();
	for (int now_r = max_indeg + 1; now_r > 0; ) {
		while (!frontier.empty()) {
			parallel_expand_reverse_frontier(
				pool, G, indeg, orientation,
				frontier, visited, new_nodes, now_low_indeg
			);
			if (!new_nodes.empty()) {
				vis_order.insert(vis_order.end(), new_nodes.begin(), new_nodes.end());
				frontier.assign(new_nodes.begin(), new_nodes.end());
			}
			else {
				frontier.clear();
			}
		}
		if (now_low_indeg >= now_r - 1) {
			find_D_number.push_back(now_r);
			int len = vis_order.size();
			for (; now_tail < len; ++now_tail) {
				int x = vis_order[now_tail];
				lock_guard<mutex> lg(idn_locks[static_cast<size_t>(x) % idn_locks.size()]);
				if (G.idn[x] < now_r) {
					G.idn[x] = now_r;
				}
			}
			now_r--;
			push_bucket_as_seeds(now_r);
		}
		else {
			while (now_r >= now_low_indeg + 2) {
				now_r--;
				push_bucket_as_seeds(now_r);
			}
		}
	}
	find_D_number.push_back(0);
}
void Graph::get_idn_loaddc() {
	memset(idn, 0, V * sizeof(int));
	ll* load = (ll*)malloc(V * sizeof(ll)), * old = (ll*)malloc(V * sizeof(ll)); memset(load, 0, V * sizeof(ll));
	int T = 500;
	for (int t = 1; t <= T; ++t) {
		if (t % 10 == 0) {
			lock_guard<mutex> lock(g_print_mutex);
			// printf("Now convex iteration number: %d\n", t);
		}
		memcpy(old, load, sizeof(ll) * V);
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			pool.spawn(0, [&, l, r] {
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (old[u] < old[v]) atomic_ref<ll>(load[u]).fetch_add(1, memory_order_relaxed);
					else atomic_ref<ll>(load[v]).fetch_add(1, memory_order_relaxed);
				}
				});
		}
		pool.wait_for_all();
	}
	free(old);

	int* indeg = (int*)malloc(V * sizeof(int)); memset(indeg, 0, V * sizeof(int));
	bool* orientation = (bool*)malloc(E * sizeof(bool));
	{
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			pool.spawn(0, [&, l, r] {
				Timer timer; timer.start();
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (load[u] < load[v]) {
						atomic_ref<ll>(load[u]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[u]).fetch_add(1, memory_order_relaxed);
						orientation[i] = false;
					}
					else {
						atomic_ref<ll>(load[v]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[v]).fetch_add(1, memory_order_relaxed);
						orientation[i] = true;
					}
				}
				});
		}
		pool.wait_for_all();
	}
	free(load);

	T = 20;
	for (int t = 1; t <= T; ++t) {
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			pool.spawn(0, [&, l, r] {
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (orientation[i] && (indeg[v] > indeg[u] + 1)) {
						atomic_ref<int>(indeg[u]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[v]).fetch_sub(1, memory_order_relaxed);
						orientation[i] = false;
					}
					else if (!orientation[i] && (indeg[u] > indeg[v] + 1)) {
						atomic_ref<int>(indeg[v]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[u]).fetch_sub(1, memory_order_relaxed);
						orientation[i] = true;
					}
				}
				});
		}
		pool.wait_for_all();
	}

	int max_indeg = 0; for (int x = 0; x < V; x++) max_indeg = max(max_indeg, indeg[x]);
	vector<int> find_D_number;
	get_D_from_orientation_parallel(*this, indeg, orientation, pool, find_D_number, max_indeg);
	printf("D_H.size = %d\n", int(find_D_number.size()));

	for (int x = 0; x < V; x++) sorted[x] = x;
	auto cmp = [this](int x, int y) { return idn[x] < idn[y]; };
	sort(sorted, sorted + V, cmp);
	int now_idn = -1;
	for (int i = 0; i < V; i++) while (idn[sorted[i]] > now_idn) position[++now_idn] = i;
	for (int idn = now_idn + 1; idn <= max_indeg; idn++) position[idn] = position[now_idn];
	position[max_indeg + 1] = V;

	vector<FlowNetwork_Div> network;
	network.reserve(NUM_THREADS);
	for (int i = 0; i < NUM_THREADS; i++) network.emplace_back(*this, orientation, indeg);
	for (int i = 0; i < NUM_THREADS; ++i) pool.spawn(0, [&, i] { network[i].init(); });
	pool.wait_for_all();

	for (int i = find_D_number.size() - 1; i >= 1; i--) {
		int k_l = find_D_number[i], k_u = find_D_number[i - 1];
		if (k_u - k_l < 2) continue;
		pool.spawn((position[k_u] - position[k_l]) * ll(k_l), [=, this, &network] { divide_and_conquer_no_construct(k_l, k_u, orientation, indeg, network); });
	}
	pool.wait_for_all();
	free(indeg), free(orientation);
}
void Graph::get_idn_cldc() {
	memset(idn, 0, V * sizeof(int));
	pool.spawn(1, [this] {
		get_core();
		degeneracy = 0; for (int x = 0; x < V; x++) degeneracy = max(degeneracy, core[x]);
		int flownetwork_cnt = 0;
		for (int p = 2; p <= degeneracy; p *= 2) flownetwork_cnt++;
		Set<int>* nodes = (Set<int>*)malloc((flownetwork_cnt + 1) * sizeof(Set<int>)); for (int i = 1; i <= flownetwork_cnt; i++) nodes[i].alloc(V);
		for (int x = 0; x < V; x++) {
			if (core[x] == 0 || core[x] == 1) idn[x] = core[x];
			else {
				int y = core[x], k = 0;
				if (y >= (1 << 16)) y >>= 16, k += 16; if (y >= (1 << 8)) y >>= 8, k += 8; if (y >= (1 << 4)) y >>= 4, k += 4; if (y >= (1 << 2)) y >>= 2, k += 2; if (y >= (1 << 1)) k += 1;
				nodes[k].insert(x);
				idn[x] = 1 << (k - 1);
			}
		}
		for (int i = 1; i <= flownetwork_cnt; i++) {
			pool.spawn(1, [=, this] { get_D_from_core(i, nodes[i]); });
		}
		});

	ll* load = (ll*)malloc(V * sizeof(ll)), * old = (ll*)malloc(V * sizeof(ll)); memset(load, 0, V * sizeof(ll));
	int T = 500;
	for (int t = 1; t <= T; ++t) {
		memcpy(old, load, sizeof(ll) * V);
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		auto counter = make_shared<atomic<int>>(0);
		auto done = make_shared<condition_variable>();
		auto mtx = make_shared<mutex>();
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			counter->fetch_add(1);
			pool.spawn(0, [&, l, r] {
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (old[u] < old[v]) atomic_ref<ll>(load[u]).fetch_add(1, memory_order_relaxed);
					else atomic_ref<ll>(load[v]).fetch_add(1, memory_order_relaxed);
				}
				if (counter->fetch_sub(1) == 1) { lock_guard<mutex> lk(*mtx); done->notify_all(); }
				});
		}
		unique_lock<mutex> lk(*mtx);
		done->wait(lk, [&] { return counter->load() == 0; });
	}
	free(old);

	int* indeg = (int*)malloc(V * sizeof(int)); memset(indeg, 0, V * sizeof(int));
	bool* orientation = (bool*)malloc(E * sizeof(bool));
	{
		auto counter = make_shared<atomic<int>>(0);
		auto done = make_shared<condition_variable>();
		auto mtx = make_shared<mutex>();
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			counter->fetch_add(1);
			pool.spawn(0, [&, l, r] {
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (load[u] < load[v]) {
						atomic_ref<ll>(load[u]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[u]).fetch_add(1, memory_order_relaxed);
						orientation[i] = false;
					}
					else {
						atomic_ref<ll>(load[v]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[v]).fetch_add(1, memory_order_relaxed);
						orientation[i] = true;
					}
				}
				if (counter->fetch_sub(1) == 1) { lock_guard<mutex> lk(*mtx); done->notify_all(); }
				});
		}
		unique_lock<mutex> lk(*mtx);
		done->wait(lk, [&] { return counter->load() == 0; });
	}
	free(load);

	T = 20;
	for (int t = 1; t <= T; ++t) {
		auto counter = make_shared<atomic<int>>(0);
		auto done = make_shared<condition_variable>();
		auto mtx = make_shared<mutex>();
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			counter->fetch_add(1);
			pool.spawn(0, [&, l, r] {
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (orientation[i] && (indeg[v] > indeg[u] + 1)) {
						atomic_ref<int>(indeg[u]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[v]).fetch_sub(1, memory_order_relaxed);
						orientation[i] = false;
					}
					else if (!orientation[i] && (indeg[u] > indeg[v] + 1)) {
						atomic_ref<int>(indeg[v]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[u]).fetch_sub(1, memory_order_relaxed);
						orientation[i] = true;
					}
				}
				if (counter->fetch_sub(1) == 1) {
					lock_guard<mutex> lk(*mtx);
					done->notify_all();
				}
				});
		}
		unique_lock<mutex> lk(*mtx);
		done->wait(lk, [&] { return counter->load() == 0; });
	}

	int max_indeg = 0; for (int x = 0; x < V; x++) max_indeg = max(max_indeg, indeg[x]);
	vector<int> find_D_number;
	get_D_from_orientation_parallel(*this, indeg, orientation, pool, find_D_number, max_indeg);
	printf("D_H.size = %d\n", int(find_D_number.size()));

	pool.wait_for_all();

	{
		int edge_chunk = max(1000, E / (NUM_THREADS * 128) + 1);
		for (ll l = 0; l < E; l += edge_chunk) {
			ll r = min(l + edge_chunk, ll(E));
			pool.spawn(0, [&, l, r] {
				Timer timer; timer.start();
				for (int i = l; i < r; ++i) {
					int u = e[i].u, v = e[i].v;
					if (orientation[i] && idn[v] > idn[u]) {
						atomic_ref<int>(indeg[u]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[v]).fetch_sub(1, memory_order_relaxed);
						orientation[i] = false;
					}
					else if (!orientation[i] && idn[u] > idn[v]) {
						atomic_ref<int>(indeg[v]).fetch_add(1, memory_order_relaxed);
						atomic_ref<int>(indeg[u]).fetch_sub(1, memory_order_relaxed);
						orientation[i] = true;
					}
				}
				});
		}
	}
	for (int x = 0; x < V; x++) sorted[x] = x;
	auto cmp = [this](int x, int y) { return idn[x] < idn[y]; };
	sort(sorted, sorted + V, cmp);
	int now_idn = -1;
	for (int i = 0; i < V; i++) while (idn[sorted[i]] > now_idn) position[++now_idn] = i;
	for (int idn = now_idn + 1; idn <= max_indeg; idn++) position[idn] = position[now_idn];
	position[max_indeg + 1] = V;

	for (int k = 1; k <= degeneracy && k < max_indeg; k *= 2) find_D_number.push_back(k);
	sort(find_D_number.begin(), find_D_number.end(), greater<int>());
	find_D_number.erase(unique(find_D_number.begin(), find_D_number.end()), find_D_number.end());
	pool.wait_for_all();

	for (int i = find_D_number.size() - 1; i >= 1; i--) {
		int k_l = find_D_number[i], k_u = find_D_number[i - 1];
		if (k_u - k_l < 2) continue;
		pool.spawn((position[k_u] - position[k_l]) * ll(k_l), [=, this] { divide_and_conquer(k_l, k_u); });
	}
	pool.wait_for_all();
	free(indeg), free(orientation);
}
void Graph::get_D_from_core(int i, Set<int>& nodes) {
	FlowNetwork network; network.construct_from_set_core(*this, nodes, 1 << (i + 1));
	network.get_D(1 << i);
	for (int ii = 0; ii < network.vis.size; ii++) {
		int x = network.new_to_ori[network.vis.element[ii]];
		lock_guard<mutex> lg(idn_locks[static_cast<size_t>(x) % idn_locks.size()]);
		idn[x] = max(idn[x], 1 << i);
	}
}
void Graph::divide_and_conquer(int k_l, int k_u) {
	int k_m = (k_l + k_u) / 2;
	FlowNetwork network;
	network.construct_from_idn(*this, k_l, k_u, true);
	network.get_D(k_m);
	for (int i = 0; i < network.vis.size; i++) idn[network.new_to_ori[network.vis.element[i]]] = max(idn[network.new_to_ori[network.vis.element[i]]], k_m);
	change_sorted(position[k_l], position[k_u], k_m);
	position[k_m] = position[k_u] - network.vis.size;

	bool spawnLeft = (k_m - k_l > 1 && position[k_m] - position[k_l] != 0);
	bool spawnRight = (k_u - k_m > 1 && position[k_u] - position[k_m] != 0);

	if (spawnRight) {
		pool.spawn((position[k_u] - position[k_m]) * ll(k_u), [=, this] { divide_and_conquer(k_m, k_u); });
	}
	if (spawnLeft) {
		pool.spawn((position[k_m] - position[k_l]) * ll(k_m), [=, this] { divide_and_conquer(k_l, k_m); });
	}
}
void Graph::divide_and_conquer_binarydc(int k_l, int k_u) {
	int now_k_l = k_l, now_k_u = k_u;
	FlowNetwork network;
	{
		Set<int> nodes; nodes.alloc(V);
		for (int i = 0; i < D[k_l].size; i++) {
			int x = D[k_l].element[i];
			if (!D[k_u].in[x]) nodes.insert(x);
		}
		network.construct_from_set(*this, nodes, true);
	}
	while (now_k_u > now_k_l) {
		int now_k_m = (now_k_u + now_k_l + 1) / 2;
		if (!have_computed[now_k_m]) {
			network.get_D(now_k_m);
			for (int i = 0; i < network.vis.size; i++) {
				idn[network.new_to_ori[network.vis.element[i]]] = max(idn[network.new_to_ori[network.vis.element[i]]], now_k_m);
				D[now_k_m].insert(network.new_to_ori[network.vis.element[i]]);
			}
			for (int i = 0; i < D[k_u].size; i++) D[now_k_m].insert(D[k_u].element[i]);
			layer_edges[now_k_m] = layer_edges[k_u];
			for (int i = 0; i < network.vis.size; i++) {
				int x = network.new_to_ori[network.vis.element[i]];
				for (int j = 0; j < undeg[x]; j++) {
					Edge& ne = e[adj[x][j]];
					int y = ne.u == x ? ne.v : ne.u;
					if (!D[k_l].in[y]) continue;
					if (D[k_u].in[y]) {
						layer_edges[now_k_m]++;
						continue;
					}
					if (!network.vis.in[network.ori_to_new[y]]) continue;
					if (x == ne.u) layer_edges[now_k_m]++;
				}
			}
			have_computed[now_k_m] = true;
		}
		if (layer_edges[k_l] - layer_edges[now_k_m] < (layer_edges[k_l] - layer_edges[k_u]) / 2)
			now_k_l = now_k_m;
		else
			now_k_u = now_k_m - 1;
	}
	int k_m = now_k_l;

	if (!have_computed[k_m]) {
		network.get_D(k_m);
		for (int i = 0; i < network.vis.size; i++) {
			idn[network.new_to_ori[network.vis.element[i]]] = max(idn[network.new_to_ori[network.vis.element[i]]], k_m);
			D[k_m].insert(network.new_to_ori[network.vis.element[i]]);
		}
		for (int i = 0; i < D[k_u].size; i++) D[k_m].insert(D[k_u].element[i]);
		layer_edges[k_m] = layer_edges[k_u];
		for (int i = 0; i < network.vis.size; i++) {
			int x = network.new_to_ori[network.vis.element[i]];
			for (int j = 0; j < undeg[x]; j++) {
				Edge& ne = e[adj[x][j]];
				int y = ne.u == x ? ne.v : ne.u;
				if (!D[k_l].in[y]) continue;
				if (D[k_u].in[y]) {
					layer_edges[k_m]++;
					continue;
				}
				if (!network.vis.in[network.ori_to_new[y]]) continue;
				if (x == ne.u) layer_edges[k_m]++;
			}
		}
	}
	if (k_m - k_l >= 2 && D[k_l].size - D[k_m].size > 0) {
		pool.spawn([this, k_l, k_m] { divide_and_conquer_binarydc(k_l, k_m); });
	}
	k_m++;
	if (!have_computed[k_m]) {
		network.get_D(k_m);
		for (int i = 0; i < network.vis.size; i++) {
			idn[network.new_to_ori[network.vis.element[i]]] = max(idn[network.new_to_ori[network.vis.element[i]]], k_m);
			D[k_m].insert(network.new_to_ori[network.vis.element[i]]);
		}
		for (int i = 0; i < D[k_u].size; i++) D[k_m].insert(D[k_u].element[i]);
		layer_edges[k_m] = layer_edges[k_u];
		for (int i = 0; i < network.vis.size; i++) {
			int x = network.new_to_ori[network.vis.element[i]];
			for (int j = 0; j < undeg[x]; j++) {
				Edge& ne = e[adj[x][j]];
				int y = ne.u == x ? ne.v : ne.u;
				if (!D[k_l].in[y]) continue;
				if (D[k_u].in[y]) {
					layer_edges[k_m]++;
					continue;
				}
				if (!network.vis.in[network.ori_to_new[y]]) continue;
				if (x == ne.u) layer_edges[k_m]++;
			}
		}
	}
	if (k_u - k_m >= 2 && D[k_m].size - D[k_u].size > 0) {
		pool.spawn([this, k_m, k_u] { divide_and_conquer_binarydc(k_m, k_u); });
	}
}
void Graph::divide_and_conquer_no_construct(int k_l, int k_u, bool* orientation, int* indeg, vector<FlowNetwork_Div>& network) {
	int k_m = (k_l + k_u) / 2;

	for (int i = position[k_l]; i < position[k_u]; i++) {
		int x = sorted[i];
		for (int j = 0; j < undeg[x]; j++) {
			Edge& ne = e[adj[x][j]];
			if (idn[ne.u] < idn[ne.v]) {
				int to = orientation[adj[x][j]] ? ne.v : ne.u;
				indeg[to]--, orientation[adj[x][j]] = false, indeg[ne.u]++;
			}
			if (idn[ne.u] > idn[ne.v]) {
				int to = orientation[adj[x][j]] ? ne.v : ne.u;
				indeg[to]--, orientation[adj[x][j]] = true, indeg[ne.v]++;
			}
		}
	}
	network[g_thread_index].get_D(k_m, k_l, k_u);
	change_sorted(position[k_l], position[k_u], k_m);
	position[k_m] = position[k_u] - network[g_thread_index].now_D_size;

	bool spawnLeft = (k_m - k_l > 1 && position[k_m] - position[k_l] != 0);
	bool spawnRight = (k_u - k_m > 1 && position[k_u] - position[k_m] != 0);

	if (spawnRight) {
		pool.spawn((position[k_m] - position[k_u]) * ll(k_u), [=, this, &network] { divide_and_conquer_no_construct(k_m, k_u, orientation, indeg, network); });
	}
	if (spawnLeft) {
		pool.spawn((position[k_l] - position[k_m]) * ll(k_m), [=, this, &network] { divide_and_conquer_no_construct(k_l, k_m, orientation, indeg, network); });
	}
}
void Graph::output_idn() {
	if (false) {
		for (int x = 0; x < V; x++) {
			printf("idn[%d] = %d\n", x, idn[x]);
		}
	}
	if (false) {
		int* cnt = (int*)malloc(V * sizeof(int));
		memset(cnt, 0, V * sizeof(int));
		int max_idn = 0;
		for (int x = 0; x < V; x++) {
			cnt[idn[x]]++; max_idn = max(max_idn, idn[x]);
		}
		for (int p = 0; p <= max_idn; p++) {
			printf("idn %d cnt = %d\n", p, cnt[p]);
		}
	}
	if (false) {
		for (int i = 0; i < V; i++) {
			int x = sorted[i];
			printf("idn[%d] = %d\n", x, idn[x]);
		}
	}
}
void Graph::change_sorted(int position_l, int position_u, int k_m) {
	position_u--;
	while (position_l <= position_u) {
		while (position_l <= position_u && idn[sorted[position_l]] < k_m) position_l++;
		while (position_l <= position_u && idn[sorted[position_u]] >= k_m) position_u--;
		if (position_l <= position_u) swap(sorted[position_l], sorted[position_u]);
	}
}
void Graph::get_core() {
	fill(core, core + V, -1);
	vector<int> p2node(V), node2p(V), d_start(V), tem_undeg(V);
	for (int i = 0; i < V; ++i) { tem_undeg[i] = undeg[i]; p2node[i] = i; }
	int max_deg = 0;
	for (int i = 0; i < V; ++i) max_deg = max(max_deg, tem_undeg[i]);
	vector<vector<int>> buckets(max_deg + 1);
	for (int i = 0; i < V; ++i) { buckets[tem_undeg[i]].push_back(i); }

	int idx = 0;
	for (int d = 0; d <= max_deg; ++d) for (int v : buckets[d]) p2node[idx++] = v;

	int nowr = -1, pointer = 0;
	while (pointer < V) {
		node2p[p2node[pointer]] = pointer;
		if (tem_undeg[p2node[pointer]] > nowr) { d_start[++nowr] = pointer; }
		else { ++pointer; }
	}

	nowr = 0;
	for (pointer = 0; pointer < V; ++pointer) {
		int now = p2node[pointer];
		if (core[now] != -1) continue;
		nowr = max(nowr, tem_undeg[now]); core[now] = nowr;
		for (int j = 0; j < undeg[now]; ++j) {
			Edge& ne = e[adj[now][j]];
			int tar = (ne.u == now) ? ne.v : ne.u;
			if (core[tar] != -1) continue;
			{
				int lp = d_start[tem_undeg[now]];
				int rp = node2p[now];
				int ln = p2node[lp];
				int rn = now;
				node2p[ln] = rp;
				node2p[rn] = lp;
				p2node[lp] = rn;
				p2node[rp] = ln;
				++d_start[tem_undeg[now]];
				--tem_undeg[now];
			}
			{
				int lp = d_start[tem_undeg[tar]];
				int rp = node2p[tar];
				int ln = p2node[lp];
				int rn = tar;
				node2p[ln] = rp;
				node2p[rn] = lp;
				p2node[lp] = rn;
				p2node[rp] = ln;
				++d_start[tem_undeg[tar]];
				--tem_undeg[tar];
			}
		}
	}
}

void FlowNetwork::construct_from_set(Graph& G, Set<int>& nodes, bool additional_indegree) {
	V = nodes.size;
	new_to_ori.alloc(V), ori_to_new.alloc(G.V), Q.alloc(V), parent.alloc(V), dist.alloc(V), cur.alloc(V), vis.alloc(V);
	undeg = (int*)malloc(V * sizeof(int)), memset(undeg, 0, V * sizeof(int)), indeg = (int*)malloc(V * sizeof(int)), memset(indeg, 0, V * sizeof(int));
	ll tem_E = 0;
	for (int i = 0; i < nodes.size; i++) {
		int x = nodes.element[i];
		ori_to_new[x] = new_to_ori.size;
		new_to_ori[new_to_ori.size] = x;
		for (int j = 0; j < G.undeg[x]; j++) {
			Edge& ne = G.e[G.adj[x][j]];
			int y = ne.u == x ? ne.v : ne.u;
			if (!nodes.in[y]) continue;
			tem_E++, undeg[ori_to_new[x]]++;
		}
	}
	tem_E /= 2, E = tem_E;
	e = (DirectedEdge*)malloc(E * sizeof(DirectedEdge)); int e_size = 0;
	for (int i = 0; i < nodes.size; i++) {
		int x = nodes.element[i];
		for (int j = 0; j < G.undeg[x]; j++) {
			Edge& ne = G.e[G.adj[x][j]];
			int y = ne.u == x ? ne.v : ne.u;
			if (!nodes.in[y]) {
				if (additional_indegree && G.idn[y] > G.idn[x]) indeg[ori_to_new[x]]++;
				continue;
			}
			if (x != ne.u) continue;
			if (indeg[ori_to_new[x]] < indeg[ori_to_new[y]]) e[e_size++] = { ori_to_new[x], ori_to_new[y], ori_to_new[x] }, indeg[ori_to_new[x]]++;
			else e[e_size++] = { ori_to_new[x], ori_to_new[y], ori_to_new[y] }, indeg[ori_to_new[y]]++;
		}
	}
	adj = (int**)malloc(V * sizeof(int*));
	for (int x = 0; x < V; x++) adj[x] = (int*)malloc(undeg[x] * sizeof(int));
	memset(undeg, 0, V * sizeof(int));
	for (int i = 0; i < E; i++) {
		DirectedEdge& ne = e[i];
		adj[ne.u][undeg[ne.u]++] = adj[ne.v][undeg[ne.v]++] = i;
	}
}
void FlowNetwork::construct_from_set_core(Graph& G, Set<int>& nodes, int additional_indegree_core_value) {
	V = nodes.size;
	new_to_ori.alloc(V), ori_to_new.alloc(G.V), Q.alloc(V), parent.alloc(V), dist.alloc(V), cur.alloc(V), vis.alloc(V);
	undeg = (int*)malloc(V * sizeof(int)), memset(undeg, 0, V * sizeof(int)), indeg = (int*)malloc(V * sizeof(int)), memset(indeg, 0, V * sizeof(int));
	E = 0;
	for (int i = 0; i < nodes.size; i++) {
		int x = nodes.element[i];
		ori_to_new[x] = new_to_ori.size;
		new_to_ori[new_to_ori.size] = x;
		for (int j = 0; j < G.undeg[x]; j++) {
			Edge& ne = G.e[G.adj[x][j]];
			int y = ne.u == x ? ne.v : ne.u;
			if (!nodes.in[y]) continue;
			E++, undeg[ori_to_new[x]]++;
		}
	}
	E /= 2;
	e = (DirectedEdge*)malloc(E * sizeof(DirectedEdge)); int e_size = 0;
	for (int i = 0; i < nodes.size; i++) {
		int x = nodes.element[i];
		for (int j = 0; j < G.undeg[x]; j++) {
			Edge& ne = G.e[G.adj[x][j]];
			int y = ne.u == x ? ne.v : ne.u;
			if (!nodes.in[y]) {
				if (G.core[y] >= additional_indegree_core_value) indeg[ori_to_new[x]]++;
				continue;
			}
			if (x != ne.u) continue;
			if (indeg[ori_to_new[x]] < indeg[ori_to_new[y]]) e[e_size++] = { ori_to_new[x], ori_to_new[y], ori_to_new[x] }, indeg[ori_to_new[x]]++;
			else e[e_size++] = { ori_to_new[x], ori_to_new[y], ori_to_new[y] }, indeg[ori_to_new[y]]++;
		}
	}
	adj = (int**)malloc(V * sizeof(int*));
	for (int x = 0; x < V; x++) adj[x] = (int*)malloc(undeg[x] * sizeof(int));
	memset(undeg, 0, V * sizeof(int));
	for (int i = 0; i < E; i++) {
		DirectedEdge& ne = e[i];
		adj[ne.u][undeg[ne.u]++] = adj[ne.v][undeg[ne.v]++] = i;
	}
}
void FlowNetwork::construct_from_idn(Graph& G, int k_l, int k_u, bool additional_indegree) {
	V = G.position[k_u] - G.position[k_l];
	new_to_ori.alloc(V), ori_to_new.alloc(G.V), Q.alloc(V), parent.alloc(V), dist.alloc(V), cur.alloc(V), vis.alloc(V);
	undeg = (int*)malloc(V * sizeof(int)), memset(undeg, 0, V * sizeof(int)), indeg = (int*)malloc(V * sizeof(int)), memset(indeg, 0, V * sizeof(int));
	E = 0;
	for (int i = G.position[k_l]; i < G.position[k_u]; i++) {
		int x = G.sorted[i];
		ori_to_new[x] = new_to_ori.size;
		new_to_ori[new_to_ori.size] = x;
		for (int j = 0; j < G.undeg[x]; j++) {
			Edge& ne = G.e[G.adj[x][j]];
			int y = ne.u == x ? ne.v : ne.u;
			if (G.idn[y] != k_l) continue;
			E++, undeg[ori_to_new[x]]++;
		}
	}
	E /= 2;
	e = (DirectedEdge*)malloc(E * sizeof(DirectedEdge)); int e_size = 0;
	for (int i = G.position[k_l]; i < G.position[k_u]; i++) {
		int x = G.sorted[i];
		for (int j = 0; j < G.undeg[x]; j++) {
			Edge& ne = G.e[G.adj[x][j]];
			int y = ne.u == x ? ne.v : ne.u;
			if (G.idn[y] != k_l) {
				if (additional_indegree && G.idn[y] > G.idn[x]) indeg[ori_to_new[x]]++;
				continue;
			}
			if (x != ne.u) continue;
			if (indeg[ori_to_new[x]] < indeg[ori_to_new[y]]) e[e_size++] = { ori_to_new[x], ori_to_new[y], ori_to_new[x] }, indeg[ori_to_new[x]]++;
			else e[e_size++] = { ori_to_new[x], ori_to_new[y], ori_to_new[y] }, indeg[ori_to_new[y]]++;
		}
	}
	adj = (int**)malloc(V * sizeof(int*));
	for (int x = 0; x < V; x++) adj[x] = (int*)malloc(undeg[x] * sizeof(int));
	memset(undeg, 0, V * sizeof(int));
	for (int i = 0; i < E; i++) {
		DirectedEdge& ne = e[i];
		adj[ne.u][undeg[ne.u]++] = adj[ne.v][undeg[ne.v]++] = i;
	}
}
bool FlowNetwork::DinicBFS() {
	bool reach_t = false;

	Q.clear(), dist.clear(), parent.clear(), cur.clear();
	for (int x = 0; x < V; x++) if (in_S(x)) dist[x] = 1, Q.push(x);

	while (!Q.empty()) {
		int x = Q.pop();
		for (int j = 0; j < undeg[x]; j++) {
			DirectedEdge& ne = e[adj[x][j]];
			if (ne.to != x) continue;
			int from = ne.u == ne.to ? ne.v : ne.u;
			if (in_T(from)) {
				reach_t = true;
			}
			if (dist.in[from]) continue;
			dist[from] = dist[x] + 1;
			Q.push(from);
		}
	}
	return reach_t;
}
bool FlowNetwork::DinicDFS(int x) {
	if (in_T(x)) {
		indeg[x]++, indeg[e[parent[x]].to]--, e[parent[x]].to = x;
		return true;
	}
	for (int& j = cur[x]; j < undeg[x]; j++) {
		DirectedEdge& ne = e[adj[x][j]];
		if (ne.to != x) continue;
		int from = ne.u == ne.to ? ne.v : ne.u;
		if ((dist[from] != dist[x] + 1) && !in_T(from)) continue;
		parent[from] = adj[x][j];
		if (DinicDFS(from)) {
			if (parent[x] == -2) {
				if (indeg[x] == pivot) return true;
				continue;
			}
			indeg[x]++, indeg[e[parent[x]].to]--, e[parent[x]].to = x;
			return true;
		}
	}
	return false;
}
void FlowNetwork::get_D(int k) {
	pivot = k - 1;

	while (DinicBFS()) {
		for (int x = 0; x < V; x++)	if (in_S(x)) parent[x] = -2, cur[x] = 0, DinicDFS(x);
	}

	Q.clear(), vis.clear();
	for (int x = 0; x < V; x++) {
		if (in_S(x))
			Q.push(x), vis.insert(x);
	}
	while (!Q.empty()) {
		int x = Q.pop();
		for (int j = 0; j < undeg[x]; j++) {
			DirectedEdge& ne = e[adj[x][j]];
			if (ne.to != x) continue;
			int from = ne.u == ne.to ? ne.v : ne.u;
			if (vis.in[from]) continue;
			Q.push(from), vis.insert(from);
		}
	}
	return;
}

void FlowNetwork_Enu::init_orientation() {
	orientation = (int*)malloc(G.E * sizeof(int));
	indeg = (int*)malloc(G.V * sizeof(int)); memset(indeg, 0, G.V * sizeof(int));
	epoch = (int16_t*)malloc(G.V * sizeof(int16_t)); memset(epoch, 0, G.V * sizeof(int16_t));
	parent = (int*)malloc(G.V * sizeof(int));
	dist = (int16_t*)malloc(G.V * sizeof(int16_t));
	cur = (int*)malloc(G.V * sizeof(int));
	Q.alloc(G.V); now_epoch = 0;
	for (int i = 0; i < G.E; i++) {
		Edge& ne = G.e[i];
		if (indeg[ne.u] < indeg[ne.v]) orientation[i] = ne.u, indeg[ne.u]++;
		else orientation[i] = ne.v, indeg[ne.v]++;
	}
}
void FlowNetwork_Enu::get_D(int k) {
	pivot = k - 1;
	while (DinicBFS()) {
		now_epoch++;
		if (G.pseudoarboricity != -1 && k > G.pseudoarboricity) return;
		for (int x = 0; x < G.V; x++) if (in_S(x)) parent[x] = -2, cur[x] = 0, epoch[x] = now_epoch, DinicDFS(x);
		if (G.pseudoarboricity != -1 && k > G.pseudoarboricity) return;
	}

	Q.clear();
	bool idn_changed = false;
	for (int x = 0; x < G.V; x++) {
		if (in_S(x)) {
			Q.push(x);
			mutex& m = idn_locks[x % idn_locks.size()];
			lock_guard<mutex> lk(m);
			if (G.idn[x] < k) G.idn[x] = k, idn_changed = true;
		}
	}
	while (!Q.empty()) {
		int x = Q.pop();
		for (int j = 0; j < G.undeg[x]; j++) {
			int eid = G.adj[x][j]; Edge& ne = G.e[eid];
			int from = orientation[eid] == ne.v ? ne.u : ne.v;
			if (from == x) continue;
			if (G.idn[from] >= k) continue;
			Q.push(from);
			mutex& m = idn_locks[from % idn_locks.size()];
			lock_guard<mutex> lk(m);
			if (G.idn[from] < k) G.idn[from] = k, idn_changed = true;
		}
	}
	if (!idn_changed) G.pseudoarboricity = k - 1;
	return;
}
bool FlowNetwork_Enu::DinicBFS() {
	int dist_t = INF;

	Q.clear(); now_epoch++;
	for (int x = 0; x < G.V; x++) if (in_S(x)) dist[x] = 1, Q.push(x), epoch[x] = now_epoch;

	bool break_loop = false;
	while (!Q.empty()) {
		int x = Q.pop();
		for (int j = 0; j < G.undeg[x]; j++) {
			int eid = G.adj[x][j]; Edge& ne = G.e[eid];
			int from = orientation[eid] == ne.v ? ne.u : ne.v;
			if (from == x) continue;
			if (in_T(from)) {
				dist_t = dist[x] + 2, break_loop = true; break;
			}
			if (epoch[from] == now_epoch) continue;
			dist[from] = dist[x] + 1; epoch[from] = now_epoch;
			Q.push(from);
		}
		if (break_loop) break;
	}
	return dist_t != INF;
}
bool FlowNetwork_Enu::DinicDFS(int x) {
	if (in_T(x)) {
		int eid = parent[x];
		Edge& ne = G.e[eid];
		indeg[x]++, indeg[orientation[eid]]--, orientation[eid] = x;
		return true;
	}
	for (int& j = cur[x]; j < G.undeg[x]; j++) {
		int eid = G.adj[x][j]; Edge& ne = G.e[eid];
		int from = orientation[eid] == ne.v ? ne.u : ne.v;
		if (from == x) continue;
		if (((epoch[from] < now_epoch - 1) || (dist[from] != dist[x] + 1)) && !in_T(from)) continue;
		parent[from] = eid;
		if (epoch[from] != now_epoch) cur[from] = 0, epoch[from] = now_epoch;
		if (DinicDFS(from)) {
			if (parent[x] == -2) {
				if (indeg[x] == pivot) return true;
				continue;
			}
			int eid = parent[x];
			Edge& ne = G.e[eid];
			indeg[x]++, indeg[orientation[eid]]--, orientation[eid] = x;
			return true;
		}
	}
	return false;
}

void FlowNetwork_Div::init() {
	parent.alloc(G.V), dist.alloc(G.V), cur.alloc(G.V), vis.alloc(G.V), Q.alloc(G.V);
}
void FlowNetwork_Div::get_D(int k, int k_l_, int k_u_) {
	pivot = k - 1, k_l = k_l_, k_u = k_u_, now_D_size = 0;
	while (DinicBFS()) {
		for (int i = G.position[k_l]; i < G.position[k_u]; i++) {
			int x = G.sorted[i];
			if (in_S(x)) parent[x] = -2, cur[x] = 0, DinicDFS(x);
		}
	}

	Q.clear();
	for (int i = G.position[k_l]; i < G.position[k_u]; i++) {
		int x = G.sorted[i];
		if (in_S(x)) {
			Q.push(x);
			G.idn[x] = k;
			now_D_size++;
		}
	}
	while (!Q.empty()) {
		int x = Q.pop();
		for (int j = 0; j < G.undeg[x]; j++) {
			int eid = G.adj[x][j]; Edge& ne = G.e[eid];
			int from = orientation[eid] ? ne.u : ne.v;
			if (from == x) continue;
			if (G.idn[from] != k_l) continue;
			lock_guard<mutex> lg(idn_locks[static_cast<size_t>(from) % idn_locks.size()]);
			G.idn[from] = k;
			Q.push(from);
			now_D_size++;
		}
	}
	return;
}
bool FlowNetwork_Div::DinicBFS() {
	int dist_t = INF;

	Q.clear(), dist.clear(), parent.clear(), cur.clear();
	for (int i = G.position[k_l]; i < G.position[k_u]; i++) {
		int x = G.sorted[i];
		if (in_S(x)) {
			dist[x] = 1;
			Q.push(x);
		}
	}

	bool break_loop = false;
	while (!Q.empty()) {
		int x = Q.pop();
		for (int j = 0; j < G.undeg[x]; j++) {
			int eid = G.adj[x][j]; Edge& ne = G.e[eid];
			int from = orientation[eid] ? ne.u : ne.v;
			check(G.idn[from] >= k_l, "from error");
			if (from == x || G.idn[from] != k_l) continue;
			if (in_T(from)) {
				dist_t = dist[x] + 2;
				break_loop = true; break;
			}
			if (dist.in[from] || G.idn[from] != k_l) continue;
			dist[from] = dist[x] + 1;
			Q.push(from);
		}
		if (break_loop) break;
	}
	return dist_t != INF;
}
bool FlowNetwork_Div::DinicDFS(int x) {
	if (in_T(x)) {
		int eid = parent[x];
		Edge& ne = G.e[eid]; int to = orientation[eid] ? ne.v : ne.u;
		indeg[x]++, indeg[to]--, orientation[eid] = !orientation[eid];
		return true;
	}
	for (int& j = cur[x]; j < G.undeg[x]; j++) {
		int eid = G.adj[x][j]; Edge& ne = G.e[eid];
		int from = orientation[eid] ? ne.u : ne.v;
		check(G.idn[from] >= k_l, "from error2");
		if (from == x || G.idn[from] != k_l) continue;
		if ((!dist.in[from] || (dist[from] != dist[x] + 1)) && !in_T(from)) continue;
		parent[from] = eid;
		if (DinicDFS(from)) {
			if (parent[x] == -2) {
				if (indeg[x] == pivot) return true;
				continue;
			}
			int eid = parent[x];
			Edge& ne = G.e[eid]; int to = orientation[eid] ? ne.v : ne.u;
			indeg[x]++, indeg[to]--, orientation[eid] = !orientation[eid];
			return true;
		}
	}
	return false;
}

int main(int argc, char** argv) {
	if (argc != 4) {
	argument_error:
		printf("Usage: ./main <dataset_address> <algorithm> <number_of_threads>\n");
		printf("algorithm:\n");
		printf("-if: IncrFlow\n");
		printf("-bd: BinaryDC\n");
		printf("-md: MeanDC\n");
		printf("-cd: CoreDC\n");
		printf("-hd: HeatDC\n");
		printf("-ch: CoreHeatDC\n");
		return 0;
	}
	Timer timer; double runtime;
	char dataset_address[256]; strcpy(dataset_address, argv[1]);
	if (strcmp(argv[2], "-if") == 0) algorithm_used = ENU;
	else if (strcmp(argv[2], "-md") == 0) algorithm_used = DC;
	else if (strcmp(argv[2], "-cd") == 0) algorithm_used = COREDC;
	else if (strcmp(argv[2], "-bd") == 0) algorithm_used = BINARYDC;
	else if (strcmp(argv[2], "-hd") == 0) algorithm_used = LOADDC;
	else if (strcmp(argv[2], "-ch") == 0) algorithm_used = CLDC;
	else goto argument_error;
	NUM_THREADS = stoi(argv[3]);

	printf("----------Density Decomposition Computation----------\n");
	printf("- %-20s: %s\n", "Dataset address", dataset_address);
	printf("- %-20s: %s\n", "Algorithm used", argv[2]);

	timer.start();
	Graph G;
	G.read_graph_from_dataset(dataset_address);
	timer.end(); runtime = timer.time();
	printf("- %-20s: %d, %d\n", "|E|, |V|", G.E, G.V);
	printf("- %-20s: %lf\n", "Read graph time", runtime);

	timer.start();
	pool.create_threads(NUM_THREADS);
	if (algorithm_used == ENU) G.get_idn_enu();
	else if (algorithm_used == DC) G.get_idn_dc();
	else if (algorithm_used == COREDC) G.get_idn_coredc();
	else if (algorithm_used == BINARYDC) G.get_idn_binarydc();
	else if (algorithm_used == LOADDC) G.get_idn_loaddc();
	else if (algorithm_used == CLDC) G.get_idn_cldc();
	timer.end(); runtime = timer.time();
	printf("- %-20s: %lf\n", "Get IDN time", runtime);

	G.output_idn();

#if defined(_WIN32)
	PROCESS_MEMORY_COUNTERS pmc; HANDLE hProcess = GetCurrentProcess();
	if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
		SIZE_T peakMemory = pmc.PeakWorkingSetSize;
		printf("- %-20s: %lf MB\n", "Peak memory usage", peakMemory / (1024.0 * 1024.0));
	}
	else { cerr << "Failed to get memory info on Windows.\n"; }
#else
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
		long peakMemoryBytes = usage.ru_maxrss; printf("- %-20s: %lf MB\n", "Peak memory usage", peakMemoryBytes / (1024.0 * 1024.0));
#else
		long peakMemoryKB = usage.ru_maxrss; printf("- %-20s: %lf MB\n", "Peak memory usage", peakMemoryKB / 1024.0);
#endif
	}
	else { cerr << "Failed to get memory info on POSIX system.\n"; }
#endif

	return 0;
}