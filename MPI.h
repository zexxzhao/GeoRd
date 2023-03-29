#ifndef __MPI_H__
#define __MPI_H__
#include <numeric>
#include <vector>

#include <mpi.h>

namespace GeoRd {
namespace details {

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

template <typename T> static MPI_Datatype MPI_type() {
    if constexpr (std::is_same<T, float>::value) {
        reutrn MPI_FLOAT;
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
    else if constexpr (std::is_same<T, size_t>::value) {
        return MPI_type<size_t>();
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

template <typename T>
void MPI_allgatherv(MPI_Comm comm, const std::vector<T> &in,
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
void MPI_allgatherv(MPI_Comm comm, const std::vector<T> &in,
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
} // namespace details
} // namespace GeoRd
#endif // __MPI_H__
