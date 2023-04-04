#include <iostream>

#include "TestMPI.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    details::init(argc, argv);
	::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
	::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
	delete listeners.Release(listeners.default_result_printer());
	delete listeners.Release(listeners.default_xml_generator());

	listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
