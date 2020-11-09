import sys
import os
from tqdm import tqdm
from queue import Queue
from multiprocessing import Pool

sys.setrecursionlimit(1000000)


class DisjointSet:
    '''
     Disjoint Set data structure (Union–Find), is a data structure that keeps track of a
     set of elements partitioned into a number of disjoint (nonoverlapping) subsets.

     Methods:
        find: Determine which subset a particular element is in. Takes an element of any
        subset as an argument and returns a subset that contains our element.

        union: Join two subsets into a single subset. Takes two elements of any subsets
        from disjoint_set and returns a disjoint_set with merged subsets.

        get: returns current disjoint set.
    '''
    
    def __init__(self, init_arr):
        self._disjoint_set = dict()
        self._groups = dict()
        self._set_size = dict()
        if init_arr:
            for item in list(set(init_arr)):
                self._disjoint_set[item] = item
                self._set_size[item] = 1
    
    def find(self, elem):
        if elem in self._disjoint_set:
            _parent = self._disjoint_set[elem]
            if elem == _parent:
                return _parent
            else:
                self._disjoint_set[elem] = self.find(_parent)
                return self._disjoint_set[elem]
        # init
        self._disjoint_set[elem] = elem
        self._set_size[elem] = 1
        return elem
    
    def union(self, elem1, elem2):
        parent_elem1 = self.find(elem1)
        parent_elem2 = self.find(elem2)
        
        if parent_elem1 != parent_elem2:
            if self._set_size[parent_elem1] < self._set_size[parent_elem2]:
                self._disjoint_set[parent_elem1] = self._disjoint_set[parent_elem2]
                self._set_size[parent_elem2] += self._set_size[parent_elem1]
                del self._set_size[parent_elem1]
            else:
                self._disjoint_set[parent_elem2] = self._disjoint_set[parent_elem1]
                self._set_size[parent_elem1] += self._set_size[parent_elem2]
                del self._set_size[parent_elem2]
    
    def get(self):
        return self._disjoint_set
    
    def length(self):
        return len(set(self._disjoint_set.values()))
    
    def group(self):
        for key, parent in self._disjoint_set.items():
            if parent not in self._groups:
                self._groups[parent] = set()
            self._groups[parent].add(key)
    
    def max(self):
        if len(self._groups) == 0:
            self.group()
        max_size = 0
        for p in self._groups:
            if len(self._groups[p]) > max_size:
                max_size = len(self._groups[p])
        return max_size


class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = set()
    
    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])
    
    def add_neighbor(self, neighbor):
        self.adjacent.add(neighbor)
    
    def get_connections(self):
        return self.adjacent
    
    def get_id(self):
        return self.id


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0
    
    def __iter__(self):
        return iter(self.vert_dict.values())
    
    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex
    
    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None
    
    def add_edge(self, frm, to):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)
        
        self.vert_dict[frm].add_neighbor(self.vert_dict[to])
        self.vert_dict[to].add_neighbor(self.vert_dict[frm])
    
    def get_vertices(self):
        return self.vert_dict.keys()


def get_adjacent_words(word_vertex, depth=3):
    # print("-----{}-----".format(id2str[v.get_id()]))
    curr_visited = set()
    _q = Queue()
    _q.put(word_vertex)
    curr_visited.add(word_vertex.get_id())
    visited_depth = dict()
    _ls = []
    _depth = 0
    while _depth < depth:
        _depth += 1
        _num = 0
        _adj_num = _q.qsize()
        while _num < _adj_num:
            _curr = _q.get()
            for _v in _curr.get_connections():
                if _v.get_id() not in curr_visited:
                    curr_visited.add(_v.get_id())
                    visited_depth[_v.get_id()] = _depth
                    _q.put(_v)
            _num += 1
    _ls.append(id2str[word_vertex.get_id()])
    for vid in curr_visited:
        if vid != word_vertex.get_id():
            _ls.append(id2str[vid] + "__" + str(visited_depth[vid]))
    return _ls


if __name__ == "__main__":
    dict_path = sys.argv[1]
    if len(sys.argv) > 2:
        d = int(sys.argv[2])
    else:
        d = 3
    myset = DisjointSet([])
    count = 0
    str2id = dict()
    _len = 0
    g = Graph()
    for filename in tqdm(os.listdir(dict_path)):
        # for line in tqdm(sys.stdin):
        src, tgt = filename.split(".")[0].split("-")
        count += 1
        if count > 100:
            break
        # print("==={}2{}===".format(src, tgt))
        with open(os.path.join(dict_path, filename), "r") as f:
            for line in f:
                try:
                    src_word, tgt_word = line.strip().split(" ")
                except:
                    src_word, tgt_word = line.strip().split("\t")
                # src
                src_final_word = src.upper() + "__" + src_word
                if src_final_word not in str2id:
                    str2id[src_final_word] = _len
                    _len += 1
                _src_id = str2id[src_final_word]
                
                # tgt
                tgt_final_word = tgt.upper() + "__" + tgt_word
                if tgt_final_word not in str2id:
                    str2id[tgt_final_word] = _len
                    _len += 1
                _tgt_id = str2id[tgt_final_word]
                
                # myset.union(src_final_word, tgt_final_word)
                g.add_edge(_src_id, _tgt_id)
    id2str = dict([(v, k) for k, v in str2id.items()])
    
    # pool = Pool(10)
    
    with open("dict.merge_dep{}.txt".format(str(d)), "w") as fw:
        for v in tqdm(g.vert_dict.values()):
        # for _id in tqdm(range(1000)):
        #     v = g.get_vertex(_id)
            _ls = get_adjacent_words(v, d)
            #     sys.stdout.write("\t".join(_ls)+"\n")
            # for _ls in list(tqdm(pool.imap(get_adjacent_words, g.vert_dict.values()))):
            fw.write("\t".join(_ls) + "\n")
    
    print("finished")
