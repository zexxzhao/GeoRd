#ifndef __MPI_H__
#define __MPI_H__
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>
#include "utils.h"
#define varname(name, line) name##line
#ifndef warning
#define warning(...)                                                           \
    do {                                                                       \
        int varname(rank, __LINE__) = details::MPI_rank(MPI_COMM_WORLD);                \
        int varname(size, __LINE__) = details::MPI_size(MPI_COMM_WORLD);                \
        fprintf(stderr,                                                        \
                "********************************************************"     \
                "***************\n");                                          \
        fprintf(stderr, "* RANK(%d/%d)\n", varname(rank, __LINE__),            \
                varname(size, __LINE__));                                      \
        fprintf(stderr, "* File(%s), Function(%s), Line(%d):\n", __FILE__,     \
                __FUNCTION__, __LINE__);                                       \
        fprintf(stderr, "* ");                                                 \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr,                                                        \
                "********************************************************"     \
                "***************\n");                                          \
    } while (0)
#endif

#ifndef error
#define error(...)                                                             \
    do {                                                                       \
        int varname(rank, __LINE__) = details::MPI_rank(MPI_COMM_WORLD);                \
        int varname(size, __LINE__) = details::MPI_size(MPI_COMM_WORLD);                \
        fprintf(stderr,                                                        \
                "********************************************************"     \
                "***************\n");                                          \
        fprintf(stderr, "* RANK(%d/%d)\n", varname(rank, __LINE__),            \
                varname(size, __LINE__));                                      \
        fprintf(stderr, "* File(%s), Function(%s), Line(%d):\n", __FILE__,     \
                __FUNCTION__, __LINE__);                                       \
        fprintf(stderr, "* ");                                                 \
        fprintf(stderr, __VA_ARGS__);                                          \
        fprintf(stderr,                                                        \
                "********************************************************"     \
                "***************\n");                                          \
        MPI_Abort(MPI_COMM_WORLD, -1);                                         \
    } while (0)
#endif


namespace GeoRd {
namespace details {

struct MPI_Base {
    MPI_Base(int argc, char **argv) {
        MPI_Init(&argc, &argv);
    }

    ~MPI_Base() {
        MPI_Finalize();
    }
};

inline void init(int argc, char **argv) {
    static MPI_Base mpi_base(argc, argv);
}

inline int MPI_rank(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

inline int MPI_size(MPI_Comm comm = MPI_COMM_WORLD) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

template <typename T> constexpr MPI_Datatype MPI_type() {
    if constexpr (std::is_same<T, float>::value) {
        return MPI_FLOAT;
    }
    else if constexpr (std::is_same<T, double>::value) {
        return MPI_DOUBLE;
    }
    else if constexpr (std::is_same<T, int>::value) {
        return MPI_INT;
    }
    else if constexpr (std::is_same<T, int64_t>::value) {
        return MPI_INT64_T;
    }
    else if constexpr (std::is_same<T, unsigned short>::value) {
        return MPI_UNSIGNED_SHORT;
    }
    else if constexpr (std::is_same<T, unsigned>::value) {
        return MPI_UNSIGNED;
    }
    else if constexpr (std::is_same<T, unsigned long>::value) {
        return MPI_UNSIGNED_LONG;
    }
    else {
        error("Unknown MPI type: %s\n", typeid(T).name());
        return 0;
    }
}

template <typename T>
void MPI_gather(MPI_Comm comm, const std::vector<T> &in, std::vector<T> &out,
                int recv_proc = 0) {
    const auto size = MPI_size(comm);
    std::vector<int> pcounts(size);
    int local_size = in.size();
    MPI_Gather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT, recv_proc,
               comm);

    std::vector<int> offsets(size + 1, 0);
    for (size_t i = 1; i <= size; ++i) {
        offsets[i] = offsets[i - 1] + pcounts[i - 1];
    }
    const size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
    out.resize(n);
    MPI_Gatherv(in.data(), in.size(), MPI_type<T>(), out.data(), pcounts.data(),
                offsets.data(), MPI_type<T>(), recv_proc, comm);
}

// For Point3D
template <typename T>
auto MPI_gather(MPI_Comm comm, const std::vector<T> &in,
                std::vector<T> &out) -> typename std::enable_if<details::Subscriptable<T>::value>::type {
    using DataType = typename T::AtomizedType;
    std::vector<DataType> in_data, out_data;
    in_data.reserve(in.size() * 3);
    for (const auto &p : in) {
        in_data.push_back(p.x);
        in_data.push_back(p.y);
        in_data.push_back(p.z);
    }
    MPI_gather(comm, in_data, out_data);
    out.reserve(out_data.size() / 3);
    for (size_t i = 0; i < out.size(); ++i) {
        out.emplace_back(out_data[i * 3], out_data[i * 3 + 1],
                         out_data[i * 3 + 2]);
    }
}

template <typename T>
void MPI_allgather(MPI_Comm comm, const std::vector<T> &in,
                    std::vector<T> &out) {
    const auto size = MPI_size(comm);
    std::vector<int> pcounts(size);
    int local_size = in.size();
    MPI_Allgather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT, comm);

    std::vector<int> offsets(size + 1, 0);
    for (size_t i = 1; i <= size; ++i) {
        offsets[i] = offsets[i - 1] + pcounts[i - 1];
    }
    const size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
    out.resize(n);
    MPI_Allgatherv(in.data(), in.size(), MPI_type<T>(), out.data(),
                   pcounts.data(), offsets.data(), MPI_type<T>(), comm);
}

template <typename T>
void MPI_allgather(MPI_Comm comm, const std::vector<T> &in,
                    std::vector<std::vector<T>> &out) {
    const auto size = MPI_size(comm);
    std::vector<int> pcounts(size);
    int local_size = in.size();
    MPI_Allgather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT, comm);

    std::vector<int> offsets(size + 1, 0);
    for (size_t i = 1; i <= size; ++i) {
        offsets[i] = offsets[i - 1] + pcounts[i - 1];
    }
    const size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
    std::vector<T> cache;
    cache.resize(n);
    MPI_Allgatherv(in.data(), in.size(), MPI_type<T>(), cache.data(),
                   pcounts.data(), offsets.data(), MPI_type<T>(), comm);

    out.resize(size);
    for (size_t i = 0; i < size; ++i) {
        auto it = cache.begin();
        out.push_back({it + offsets[i], it + offsets[i + 1]});
    }
}

template <typename T>
void MPI_broadcast(MPI_Comm comm, std::vector<T> &value, int send_proc = 0) {
    size_t bsize = value.size();
    MPI_Bcast(&bsize, 1, MPI_UNSIGNED, send_proc, comm);
    value.resize(bsize);
    MPI_Bcast(value.data(), value.size(), MPI_type<T>(), send_proc, comm);
}

template <typename T>
void MPI_scatter(MPI_Comm comm, const std::vector<std::vector<T>> send_data,
                 std::vector<T> recv_data, int send_proc = 0) {
    const auto size = MPI_size(comm);
    std::vector<int> pcounts(size);
    for (size_t i = 0; i < size; ++i) {
        pcounts[i] = send_data[i].size();
    }
    MPI_broadcast(comm, pcounts, send_proc);

    std::vector<int> offsets(size + 1, 0);
    for (size_t i = 1; i <= size; ++i) {
        offsets[i] = offsets[i - 1] + pcounts[i - 1];
    }
    const size_t n = std::accumulate(pcounts.begin(), pcounts.end(), 0);
    std::vector<T> cache;
    cache.resize(n);
    for (size_t i = 0; i < size; ++i) {
        std::copy(send_data[i].begin(), send_data[i].end(),
                  cache.begin() + offsets[i]);
    }
    recv_data.resize(pcounts[MPI_rank(comm)]);
    MPI_Scatterv(cache.data(), pcounts.data(), offsets.data(), MPI_type<T>(),
                 recv_data.data(), recv_data.size(), MPI_type<T>(), send_proc,
                 comm);
}

template <typename T, typename X>
T all_reduce(MPI_Comm comm, const T &value, X op) {
    T out;
    MPI_Allreduce(const_cast<T *>(&value), &out, 1, MPI_type<T>(), op, comm);
    return out;
}

template <typename T> T MPI_sum(MPI_Comm comm, const T &v) {
    return all_reduce(comm, v, static_cast<MPI_Op>(MPI_SUM));
}

template <typename T> T MPI_max(MPI_Comm comm, const T &v) {
    return all_reduce(comm, v, static_cast<MPI_Op>(MPI_MAX));
}

template <typename T> T MPI_min(MPI_Comm comm, const T &v) {
    return all_reduce(comm, v, static_cast<MPI_Op>(MPI_MIN));
}

template <typename T, typename X>
void all_reduce(MPI_Comm comm, const std::vector<T> &in, std::vector<T> &out,
                X op) {
    out.resize(in.size());
    if (MPI_size(comm) > 1) {
        MPI_Allreduce(const_cast<T *>(in.data()), out.data(), in.size(),
                      MPI_type<T>(), op, comm);
    }
    else {
        std::copy(in.begin(), in.end(), out.begin());
    }
}

template <typename T>
void MPI_sum(MPI_Comm comm, const std::vector<T> &in, std::vector<T> &out) {
    all_reduce(comm, in, out, static_cast<MPI_Op>(MPI_SUM));
}

template <typename T>
void MPI_max(MPI_Comm comm, const std::vector<T> &in, std::vector<T> &out) {
    all_reduce(comm, in, out, static_cast<MPI_Op>(MPI_MAX));
}

template <typename T>
void MPI_min(MPI_Comm comm, const std::vector<T> &in, std::vector<T> &out) {
    all_reduce(comm, in, out, static_cast<MPI_Op>(MPI_MIN));
}

template <typename T>
void MPI_alltoall(MPI_Comm comm, std::vector<std::vector<T>> &in,
                  std::vector<std::vector<T>> &out) {
    const size_t size = MPI_size(comm);

    assert(in.size() == size);
    std::vector<int> data_size_send(size);
    std::vector<int> data_offset_send(size + 1, 0);
    for (int p = 0; p < size; p++) {
        data_size_send[p] = in[p].size();
        data_offset_send[p + 1] = data_offset_send[p] + data_size_send[p];
    }

    std::vector<int> data_size_recv(size);
    MPI_Alltoall(data_size_send.data(), 1, MPI_type<int>(),
                 data_size_recv.data(), 1, MPI_type<int>(), comm);

    std::vector<int> data_offset_recv(size + 1, 0);
    std::vector<T> data_send(data_offset_send[size]);
    for (int p = 0; p < size; p++) {
        data_offset_recv[p + 1] = data_offset_recv[p] + data_size_recv[p];
        std::copy(in[p].begin(), in[p].end(),
                  data_send.begin() + data_offset_send[p]);
    }

    std::vector<T> data_recv(data_offset_recv[size]);
    MPI_Alltoallv(data_send.data(), data_size_send.data(),
                  data_offset_send.data(), MPI_type<T>(), data_recv.data(),
                  data_size_recv.data(), data_offset_recv.data(), MPI_type<T>(),
                  comm);

    out.resize(size);
    for (int p = 0; p < size; p++) {
        out[p].resize(data_size_recv[p]);
        std::copy(data_recv.begin() + data_offset_recv[p],
                  data_recv.begin() + data_offset_recv[p + 1], out[p].begin());
    }
}

template <typename T>
void MPI_alltoall(MPI_Comm comm, const std::vector<std::vector<T>> &in,
                  std::vector<T> &out) {
    const size_t size = MPI_size(comm);

    assert(in.size() == size);
    std::vector<int> data_size_send(size);
    std::vector<int> data_offset_send(size + 1, 0);
    for (int p = 0; p < size; p++) {
        data_size_send[p] = in[p].size();
        data_offset_send[p + 1] = data_offset_send[p] + data_size_send[p];
    }

    std::vector<int> data_size_recv(size);
    MPI_Alltoall(data_size_send.data(), 1, MPI_type<int>(),
                 data_size_recv.data(), 1, MPI_type<int>(), comm);

    std::vector<int> data_offset_recv(size + 1, 0);
    std::vector<T> data_send(data_offset_send[size]);
    for (int p = 0; p < size; p++) {
        data_offset_recv[p + 1] = data_offset_recv[p] + data_size_recv[p];
        std::copy(in[p].begin(), in[p].end(),
                  data_send.begin() + data_offset_send[p]);
    }

    auto &data_recv = out;
    data_recv.resize(data_offset_recv[size]);
    MPI_Alltoallv(data_send.data(), data_size_send.data(),
                  data_offset_send.data(), MPI_type<T>(), data_recv.data(),
                  data_size_recv.data(), data_offset_recv.data(), MPI_type<T>(),
                  comm);
}

template <
    typename Key, typename Value, typename Hash = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>>
struct MPI_DataDistributor {
    static_assert(std::is_arithmetic<Key>::value and std::is_arithmetic<Value>::value, "Key and Value must be arithmetic types.");
    template <typename T = Value>
    using Map = std::unordered_map<Key, T, Hash, KeyEqual>;

    int root;
    MPI_Comm comm;
    std::vector<std::vector<int>> rank;
    std::vector<Key> keys;
    std::vector<Value> values;
    Map<std::vector<int>> ranks;

    MPI_DataDistributor(MPI_Comm comm, int root = 0) : root(root), comm(comm) {}

    void gather(const std::vector<Key> &keys_send,
                const std::vector<Value> &values_send) {
        assert(keys.size() == values.size());
        const int size = MPI_size(comm);

        std::vector<Key> keys_recv;
        std::vector<Value> values_recv;
        std::vector<int> rank_send(keys_send.size(), MPI_rank(comm));
        std::vector<int> rank_recv;
        MPI_gather(comm, keys_send, keys_recv, root);
        MPI_gather(comm, values_send, values_recv, root);
        MPI_gather(comm, rank_send, rank_recv, root);

        if (MPI_rank(comm) == root) {
            Map<Value> map;
            assert(keys_recv.size() == values_recv.size());
            for (size_t i = 0; i < keys_recv.size(); i++) {
                auto it = map.find(keys_recv[i]);
                if (it == map.end()) {
                    map[keys_recv[i]] = values_recv[i];
                }
                else if (it->second != values_recv[i]) {
                    std::cerr << "Error: key " << keys_recv[i]
                              << " has different values on different processes"
                              << std::endl;
                    std::exit(1);
                }
                ranks[keys_recv[i]].push_back(rank_recv[i]);
            }
            // copy map to keys and values
            keys.clear();
            keys.reserve(map.size());
            values.clear();
            values.reserve(map.size());
            for (auto &kv : map) {
                keys.push_back(kv.first);
                values.push_back(kv.second);
            }
        }
    }

    void scatter(std::vector<Key> &keys_recv,
                 std::vector<Value> &values_recv) const {
        keys_recv.clear();
        values_recv.clear();
        const int size = MPI_size(comm);
        std::vector<std::vector<Key>> keys_send;
        std::vector<std::vector<Value>> values_send;
        if (MPI_rank(comm) == root) {
            keys_send.resize(size);
            values_send.resize(size);
            // assemble keys and values for each processer
            for (size_t i = 0; i < keys.size(); i++) {
                for (int r : ranks.at(keys[i])) {
                    keys_send[r].push_back(keys[i]);
                    values_send[r].push_back(values[i]);
                }
            }
        }
        MPI_scatter(comm, keys_send, keys_recv, root);
    }
};

} // namespace details
} // namespace GeoRd

#endif // __MPI_H__
