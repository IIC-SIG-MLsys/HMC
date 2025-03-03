/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include <coll.h>
#include "utils/proto.h"

#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netdb.h>

namespace hddt {

MPIOOB::MPIOOB() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}
MPIOOB::~MPIOOB() {
    MPI_Finalize();
}

RankInfoCollection MPIOOB::collectRankInfo() {
    RankInfoCollection rankCollection;
    int name_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Get_processor_name(hostname, &name_len);
    std::string ip_address = getLocalIP();

    RankInfo myInfo;
    myInfo.set_rank(rank);
    myInfo.set_hostname(hostname);
    myInfo.set_ip_address(ip_address);
    myInfo.set_timestamp(time(nullptr));

    std::string serializedData = ProtoUtils::serializeToString(myInfo);
    int data_size = serializedData.size();

    if (rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            MPI_Status status;
            MPI_Recv(&data_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            std::vector<char> buffer(data_size);
            MPI_Recv(buffer.data(), data_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);

            RankInfo receivedRank;
            ProtoUtils::deserializeFromString(receivedRank, std::string(buffer.begin(), buffer.end()));
            *rankCollection.add_infos() = receivedRank;
        }
        *rankCollection.add_infos() = myInfo;
    } else {
        MPI_Send(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(serializedData.c_str(), data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    return rankCollection;
}

void MPIOOB::distributeTask() {
    if (rank == 0) {
        ComputationGraph compGraph;
        compGraph.set_graph_version(1);
        
        ComputationGraph::RankAssignment* assignment = compGraph.add_assignments();
        assignment->set_rank(1);
        TaskChain* task_chain = assignment->add_chains();
        task_chain->set_chain_id(100);
        task_chain->set_priority(1);
        
        Task* task = task_chain->add_tasks();
        task->mutable_compute_task()->set_op_type(ComputeTask_OperatorType_REDUCE);
        task->mutable_compute_task()->set_ptr_offset(0);
        task->mutable_compute_task()->set_data_size(1024);
        task->mutable_compute_task()->set_dtype("float32");
        task->set_estimated_duration(10);

        std::string taskData = ProtoUtils::serializeToString(compGraph);
        int taskSize = taskData.size();

        for (int i = 1; i < world_size; ++i) {
            MPI_Send(&taskSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(taskData.c_str(), taskSize, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // 任务接收
        MPI_Status status;
        int data_size;
        MPI_Recv(&data_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std::vector<char> taskBuffer(data_size);
        MPI_Recv(taskBuffer.data(), data_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);

        ComputationGraph receivedGraph;
        ProtoUtils::deserializeFromString(receivedGraph, std::string(taskBuffer.begin(), taskBuffer.end()));

        std::cout << "Rank " << rank << " received ComputationGraph version: "
                  << receivedGraph.graph_version() << std::endl;
    }
}

std::string MPIOOB::getLocalIP() {
    struct ifaddrs *ifaddr, *ifa;
    int family, s;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        std::cerr << "Error: getifaddrs failed." << std::endl;
        return "";
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        family = ifa->ifa_addr->sa_family;

        if (family == AF_INET || family == AF_INET6) {
            s = getnameinfo(ifa->ifa_addr,
                            (family == AF_INET) ? sizeof(struct sockaddr_in)
                                                : sizeof(struct sockaddr_in6),
                            host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
            if (s != 0) {
                std::cerr << "Error: getnameinfo() failed: " << gai_strerror(s) << std::endl;
                continue;
            }

            // 跳过回环地址
            if (strcmp(host, "127.0.0.1") == 0 || strcmp(host, "::1") == 0) continue;

            freeifaddrs(ifaddr);
            return std::string(host);
        }
    }

    freeifaddrs(ifaddr);
    return "";
}

}
