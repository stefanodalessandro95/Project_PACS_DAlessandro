CXX       = g++
CXXFLAGS  = -std=c++11
CPPFLAGS  = -I$(mkEigenInc)  -I../Headers/opennn_headers
EXEC      = main_1D main_2D main_NN
OBJECTS_1D_2D   = FunctionData1D.o MinimizationAlgorithm.o StochGradDesc.o \
EntropySGD.o Heat.o GradientDescent.o
OBJECTS_NN 	= FunctionOnNeuralNetwork.o MinimizationAlgorithm.o StochGradDesc.o \
EntropySGD.o Heat.o
LDLIBS_EIGEN      = -L$(mkEigenLib) -leigen
LDLIBS_OPENNN			= ../Libraries/libopennn.a


.PHONY : all clean distclean

all : $(EXEC)

%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $^

main_1D : % : main_1D.o $(OBJECTS_1D_2D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDLIBS_EIGEN)

main_2D : % : main_2D.o $(OBJECTS_1D_2D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDLIBS_EIGEN)

main_NN : % : main_NN.o $(OBJECTS_NN)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDLIBS_EIGEN) $(LDLIBS_OPENNN)

clean :
	$(RM) *.o

distclean : clean
	$(RM) $(EXEC)
