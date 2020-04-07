build:
	nvcc -arch=sm_30 mat-transpose.cu -o mat-transpose

test: build
	./mat-transpose
