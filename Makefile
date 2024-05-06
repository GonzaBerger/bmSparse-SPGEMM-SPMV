#INCLUDECUDA = -I/usr/local/cuda-7.0/samples/common/inc/
# HEADERNVMLAPI = -L/usr/lib64/nvidia -lnvidia-ml -L/usr/lib64 -lcuda -I/usr/include -lpthread

#compilers
#CC=gcc
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE = float

#CUDA_PARAMETERS
# NVCC_FLAGS = -g -O3 -w -m64 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30
# NVCC_FLAGS = -O3 -w -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52

<<<<<<< HEAD
NVCC_FLAGS = -O3 -w -arch=sm_86 -Xptxas -dlcm=ca --expt-extended-lambda --expt-relaxed-constexpr #--expt-extended-lambda
=======
NVCC_FLAGS = -arch=sm_86 -Xptxas -dlcm=ca --expt-extended-lambda --expt-relaxed-constexpr #--expt-extended-lambda
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
#NVCC_FLAGS = -O3 -w -m64 -arch=sm_30

#NVCC_FLAGS = -Xcompiler -fopenmp -O3 -w -m64 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -Xptxas -dlcm=cg
#NVCC_FLAGS = -O3 -std=c99 -w -m64

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda
CUSP_PATH = ./cusp
#MKLROOT = /opt/intel/mkl
MKLROOT = /home/gpgpu/software/mkl
#includes
#INCLUDES = -I./include -I$(CUSP_PATH)/
INCLUDES = -I$(CUSP_PATH)/ -I$(CUDA_INSTALL_PATH)/include -I./include -I${MKLROOT}/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
MKL_LIBS =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -ltbb -lstdc++ -lpthread -lm -ldl
MKL_LIBS = -lpthread -lm# -L/opt/intel/lib/intel64_lin ${MKLROOT}/lib/intel64_lin/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64_lin/libmkl_intel_thread.a ${MKLROOT}/lib/intel64_lin/libmkl_core.a -lstdc++ -lpthread -lm -ldl -liomp5
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib  -lcudart -lcuda -lcusparse -lnvidia-ml # -lgomp
#LIBS = $(MKL_LIBS)
LIBS = $(CUDA_LIBS) $(CLANG_LIBS) $(MKL_LIBS)
#options
#OPTIONS = -std=c99


#FILES = main.c 
 
<<<<<<< HEAD
FILES_SPGEMM = src/bmSparse_SPGEMM.cu \
=======
FILES = src/bmSparse.cu \
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
	src/bmSpMatrix.cu \
        src/cuSparse_mult.cu \
	src/cuSparse_spmv.cu \
        src/reader.cu \
	#src/blkDensity.cu

<<<<<<< HEAD
FILES_SPMV = src/bmSparse_SPMV.cu \
			 src/bmSpMatrix.cu \
			 src/cuSparse_spmv.cu \
        	 src/reader.cu \

FILES_BMSPARSE = src/bmSparse.cu \
	src/bmSpMatrix.cu \
	src/cuSparse_spmv.cu \
	src/reader.cu \

spgemm:
	$(CC) $(NVCC_FLAGS) $(FILES_SPGEMM) -o bmsparse_spgemm_$(VALUE_TYPE) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__

spmv:
	$(CC) $(NVCC_FLAGS) $(FILES_SPMV) -o bmsparse_spmv_$(VALUE_TYPE) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__

bmsparse:
	$(CC) $(NVCC_FLAGS) $(FILES_BMSPARSE) -o bmsparse_$(VALUE_TYPE) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__
=======

make:
	#$(CC) $(NVCC_FLAGS) $(FILES) -o bmsparse_$(VALUE_TYPE) $(INCLUDES) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__
	$(CC) $(NVCC_FLAGS) $(FILES) -o bmsparse_$(VALUE_TYPE) $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) -D__$(VALUE_TYPE)__
>>>>>>> 7a2e3211c04a7e2059e8279160648d6e25bce12e
