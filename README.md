# Scaling Up Density Decomposition at Billion-Scale Graphs

# Datasets

Because some datasets used in the paper are too large to be uploaded to GitHub, we have summarized the download links for the dataset in the table below.

| Dataset | Link |
| --- | --- |
| LiveJ | https://law.di.unimi.it/webdata/ljournal-2008/ |
| HW09 | https://law.di.unimi.it/webdata/hollywood-2009/ |
| DBpedia | http://www.konect.cc/networks/dbpedia-link/ |
| P2P | https://networkrepository.com/tech-p2p.php |
| WikiEn | https://networkrepository.com/web-wikipedia-link-en13-all.php |
| CC12 | https://networkrepository.com/web-cc12-hostgraph.php |
| ITAll | https://networkrepository.com/web-it-2004-all.php |
| Twitter | https://networkrepository.com/soc-twitter.php |
| GSH | https://law.di.unimi.it/webdata/gsh-2015/ |
| SKAll | https://networkrepository.com/web-sk-2005-all.php |

# Preprocess

The dataset file needs to be preprocessed as the format below, where u_i and v_i are two endpoints of an undirected edge (u_i, v_i).

```
<|V|> <|E|>
<u_1> <v_1>
<u_2> <v_2>
<u_3> <v_3>
...
```

# Usage

Compile the program using a C++20-compliant compiler:

```
g++ main.cpp -o main -std=c++20 -O3 -pthread
```

Run the executable with the following arguments:

```
./main <dataset_address> <algorithm> <thread_number>
```

### Arguments

* **`dataset_address`**
  Path to the input dataset file.

* **`algorithm`**
  Specifies which algorithm to run. The available options are:

  |  Flag | Algorithm Name | Description                           |
  | ----: | -------------- | ------------------------------------- |
  | `-if` | IncrFlow       | Incremental Flow–based algorithm      |
  | `-bd` | BinaryDC       | Binary-based Density Clustering       |
  | `-md` | MeanDC         | Mean-based Density Clustering         |
  | `-cd` | CoreDC         | Core-based Density Clustering         |
  | `-hd` | HeatDC         | Heat-based Density Clustering         |
  | `-ch` | CoreHeatDC     | Hybrid Core + Heat Density Clustering |

* **`number_of_threads`**
  Number of threads to use for parallel execution (uses POSIX threads).

### Example

```bash
./main graphs/example_graph.txt -cd 8
```

This command runs the **CoreDC** algorithm on `graphs/example_graph.txt` using **8 threads**.
