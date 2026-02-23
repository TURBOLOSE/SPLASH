.PHONY: clean time

NAME = test

MAIN_DEPEND = test.cpp src/geometry/sph_gen.h src/MUSCL_base/MUSCL_base.hpp \
 src/geometry/MUSCL_geometry.hpp $(wildcard src/Riemann_solvers/*.hpp) $(wildcard src/physics/*.hpp)

SOURCE_FILES = test.cpp \
 src/pmp/surface_mesh.cpp \
 src/pmp/algorithms/subdivision.cpp \
 src/pmp/algorithms/differential_geometry.cpp

OBJ_FILES = $(SOURCE_FILES:.cpp=.o)

CXX = icpx
CXXFLAGS = -std=c++23 -Ofast -qopenmp -xhost
LDFLAGS = 

$(NAME): $(OBJ_FILES) $(NAME).o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(NAME) $(OBJ_FILES)

$(NAME).o: $(MAIN_DEPEND)
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/pmp/surface_mesh.o: src/pmp/surface_mesh.h
src/pmp/algorithms/subdivision.o: src/pmp/algorithms/subdivision.h
src/pmp/algorithms/differential_geometry.o: src/pmp/algorithms/differential_geometry.h

clean:
	rm -f $(NAME)
	find . -name "*.o" -delete

time:
	time ./$(NAME)
