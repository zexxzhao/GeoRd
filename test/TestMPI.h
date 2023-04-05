#ifndef __TEST_MPI_H__
#define __TEST_MPI_H__

#include "../include/GeoRd.h"
#include "gtest-mpi-listener.h"

using namespace GeoRd;

TEST(MPI, Basic) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();

    int *sendbuf = new int[size];
    int *recvbuf = new int[size];

    for (int i = 0; i < size; ++i) {
        sendbuf[i] = rank;
    }

    MPI_Allgather(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(recvbuf[i], i);
    }

    delete[] sendbuf;
    delete[] recvbuf;
}

TEST(MPI, DataType) {
    ASSERT_EQ(MPI_INT, details::MPI_type<int>());
    ASSERT_EQ(MPI_DOUBLE, details::MPI_type<double>());
    ASSERT_EQ(MPI_FLOAT, details::MPI_type<float>());
    ASSERT_EQ(MPI_INT64_T, details::MPI_type<int64_t>());
    ASSERT_EQ(MPI_UNSIGNED_LONG, details::MPI_type<unsigned long>());
    ASSERT_EQ(MPI_UNSIGNED, details::MPI_type<unsigned>());
    ASSERT_EQ(MPI_UNSIGNED_SHORT, details::MPI_type<unsigned short>());
}

TEST(MPI, gather) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();

    std::vector<int> sendbuf(rank + 1);
    std::vector<int> recvbuf(size * (size + 1) / 2);
    std::fill(sendbuf.begin(), sendbuf.end(), rank);

    int root = 0;
    details::MPI_gather(MPI_COMM_WORLD, sendbuf, recvbuf, root);

    if (rank != root) {
        return;
    }
    int start = 0;
    int stop = 0;
    for (int i = 0; i < size; ++i) {
        start += i;
        stop += i + 1;
        for (int j = start; j < stop; ++j) {
            ASSERT_EQ(recvbuf[j], i);
        }
    }
}

TEST(MPI, scatter) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();

    std::vector<std::vector<int>> sendbuf(size);
    std::vector<int> recvbuf(rank + 1);

    for (int i = 0; i < size; ++i) {
        sendbuf[i].resize(i + 1);
        std::fill(sendbuf[i].begin(), sendbuf[i].end(), i);
    }

    int root = 0;
    details::MPI_scatter(MPI_COMM_WORLD, sendbuf, recvbuf, root);
    for (int i = 0; i < recvbuf.size(); ++i) {
        ASSERT_EQ(recvbuf[i], rank);
    }
}

TEST(MPI, Distributor) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();

    int root = 0;
    details::MPI_DataDistributor<int, int> distributor(MPI_COMM_WORLD, root);
    std::vector<int> keys_send(size);
    std::vector<int> value_send(size);
    auto hash = [rank](int key) -> std::size_t {
        return (((key + 7) * key) << 2) ^ (key + 31);
    };
    for (int i = 0; i < size; ++i) {
        auto key = i + rank;
        keys_send[i] = key;
        value_send[i] = hash(key);
    }
    distributor.gather(keys_send, value_send);
    if (rank == root) {
        for (int i = 0; i < distributor.keys.size(); ++i) {
            auto key = distributor.keys[i];
            ASSERT_EQ(distributor.values[i], hash(key));
            // printf("data[%d] = %d (%zu)\n", key, distributor.values[i],
            // hash(key));
        }
    }
    auto hash2 = [rank](int key) -> std::size_t {
        return ((2 * key + 7) << 2) & (key + 31);
    };
    if (rank == root) {
        for (int i = 0; i < distributor.keys.size(); ++i) {
            auto key = distributor.keys[i];
            distributor.values[i] = hash2(key);
        }
    }
    std::vector<int> key_recv;
    std::vector<int> value_recv;
    distributor.scatter(key_recv, value_recv);
    for (int i = 0; i < key_recv.size(); ++i) {
        auto key = key_recv[i];
        ASSERT_EQ(value_recv[i], hash2(key));
    }
}

#endif // __TEST_MPI_H__