#ifndef __TEST_MPI_H__
#define __TEST_MPI_H__

#include "gtest-mpi-listener.h"
#include "../include/GeoRd.h"

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

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            ASSERT_EQ(recvbuf[i * (i + 1) / 2 + j], j);
        }
    }
}

TEST(MPI, scatter) {
    int rank = details::MPI_rank();
    int size = details::MPI_size();

    std::vector<std::vector<int>> sendbuf(size);
    std::vector<int> recvbuf(size);

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

    details::MPI_DataDistributor<int, int> distributor(MPI_COMM_WORLD);
    std::vector<int> keys_send(size);
    std::vector<int> valye_send(size);
    for(int i = 0; i < size; ++i) {
        keys_send[i] = i + rank;
        valye_send[i] = i + rank;
    }
    distributor.gather(keys_send, valye_send);
    if(rank == 0) {
        for(int i = 0; i < distributor.keys.size(); ++i) {
            ASSERT_EQ(distributor.keys[i], i);
            ASSERT_EQ(distributor.values[i], i);
        }
    }
}

#endif // __TEST_MPI_H__