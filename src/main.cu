#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <sys/mman.h>
using namespace std;

#define FNNpath "../neuralData/FNN"
#define backupPath "../neuralData/backup"

#define ReLUalpha 0.1
#define randomNRuns 1
#define maxMoves 230
#define learningStep -0.01
#define totGames 20
#define costTH 2000.0
#define FLT_MAX (1.0/0.0)
#define L2Alpha 0.00
#define beta1 0.9
#define beta2 0.999
#define epsilon 0.0000001
#define certaintyP 1.1
#define verificationGames 20

extern "C" void* readF(const char* filename, int elemSize, long* nElWriteB);
extern "C" void writeF(void* d1, int size, int elSize, const char* filepath);

#define maxCacheSize 1024

__global__ void VecCMult(float* A, float* B, float c, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    B[i] = A[i]*c;
}

__global__ void MatVecMultFusedAddAct(float* a, float* b, float* c, float* addV, int Arows, int AcBr, char skAc = 1){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tileSize = blockDim.x;
    int ctx = tileSize*bx + tx;

    if(ctx >= Arows){
        return;
    }

    __shared__ float B[maxCacheSize];

    float tempVal = 0;
    
    for(int i = 0; i < AcBr; i += tileSize){
        if(tx < tileSize && tx+i < AcBr){
            B[tx] = b[tx + i];
        }
        else{
            B[tx] = 0.0;
        }
        __syncthreads();

        for(int k = 0; k < tileSize; k++){
            tempVal += a[ctx + Arows*(k+i)]*B[k];
        }
        __syncthreads();
    }
    
    //for(int i = 0; i < AcBr; i++){
    //    tempVal += a[ctx + Arows*i]*b[i];
    //}
    tempVal += addV[ctx];
    if(tempVal < 0 && skAc){
        tempVal *= ReLUalpha;
    }
    //printf("%f\n", tempVal);
    c[ctx] = tempVal;
}

__global__ void biasAddNActivationF(float* A, float* B, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float retV = A[i]+B[i];
    if(retV < 0){
        A[i] = retV*ReLUalpha;
        return;
    }
    A[i] = retV;
}
__global__ void MatCMultFusedAdd(float* A, float* B, float* C, float d, int iSize, int kSize){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    if(i >= iSize || k >= kSize){
        return;
    }
    C[k*iSize + i] = A[i] + (B[k*iSize + i]*d);
}
__global__ void VecAdd(float* A, float* B, float* C, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    C[i] = A[i]+B[i];
}
//reminder that NN is the offseted NN not the original one
__global__ void finalLayerGradient(float* NN, float* lastLLayerOutput, float* C, float* gradientWriteBack, int lastLayerSize, int penLayerSize, float* dHolder){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= lastLayerSize || k >= penLayerSize+1){
        return;
    }
    float relatedSm = lastLLayerOutput[penLayerSize + i];
    float dfx = 0;
    if(i < 64){
        for(int t = 0; t < 64; t++){
            if(t == i){
                dfx -= C[t]*(1.0 - relatedSm);
            }
            else{
                dfx += C[t]*relatedSm;
            }
        }
        dHolder[i] = dfx;
    }
    else{
        for(int t = 64; t < 128; t++){
            if(t == i){
                dfx -= C[t]*(1.0 - relatedSm);
            }
            else{
                dfx += C[t]*relatedSm;
            }
        }
        dHolder[i] = dfx;
    }

    if(k < penLayerSize){
        gradientWriteBack[k + i*penLayerSize] = dfx*lastLLayerOutput[k] + NN[k + i*penLayerSize]*L2Alpha;
        return;
    }
    gradientWriteBack[penLayerSize*lastLayerSize + i] = dfx + NN[penLayerSize*lastLayerSize + i]*L2Alpha;

}

__global__ void computeLastTheta(float* NN, float* C, float* thetaWriteBack, int iSize, int eSize){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int e = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= iSize || e >= eSize){
        return;
    }

    thetaWriteBack[i + e*iSize] = C[i]*NN[e + i*eSize];
}

__global__ void layerTolayerGradient(float* NN, float* penLayerOutput, float* theta, float* thetaWriteBack, float* gradientWriteBack, int lastLayerSize, int penLayerSize){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= lastLayerSize || k >= penLayerSize+1){
        return;
    }

    float thetaW = theta[i];
    if(penLayerOutput[i+penLayerSize] < 0.0){
        thetaW *= ReLUalpha;
    }

    if(k < penLayerSize){
        thetaWriteBack[i + k*lastLayerSize] = thetaW*NN[k + i*penLayerSize];
        gradientWriteBack[k + i*penLayerSize] = thetaW*penLayerOutput[k] + NN[k + i*penLayerSize]*L2Alpha;
        return;
    }
    gradientWriteBack[penLayerSize*lastLayerSize + i] = thetaW + NN[penLayerSize*lastLayerSize + i]*L2Alpha;
}

__global__ void sumReduction(float* A, float* writeBack, int eSize, int rSize, int sizeDBBlock){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int bi = blockIdx.x;

    if(i >= eSize || k >= rSize){
        return;
    }

    __shared__ float l1A[maxCacheSize];
    l1A[tx + ty*bx] = A[i + k*eSize];
    __syncthreads();

    int p0 = tx*2;
    int p1 = p0+1;
    float* l1AM = l1A+ty*bx;
    for(int j = 0; j < bx; j++){
        if(p1 >= bx){
            if(p0 == 0){
                writeBack[bi + k*sizeDBBlock] = l1AM[0];
            }
            return;
        }
        l1AM[p0] = l1AM[p0]+l1AM[p1];
        p0 *= 2;
        p1 *= 2;
        __syncthreads();
    }
}

__global__ void initiateToC(float* A, float b, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    A[i] = b;
}

__global__ void generateDistribution(float* A, float* retV){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    if(i >= 64 || k >= 64){
        return;
    }
    __shared__ float A1[128];
    if(threadIdx.x < 8 && threadIdx.y < 16){
        A1[threadIdx.x + threadIdx.y*8] = A[threadIdx.x + threadIdx.y*8];
        //printf("%f\n", A[threadIdx.x + threadIdx.y*8]);
    }
    __syncthreads();
    retV[i + k*64] = A1[i]*A1[k+64];
}

__global__ void expSoftMax(float* A, float* B, float maxV){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= 4096){
        return;
    }
    B[i] = exp(A[i]-maxV);
}

__global__ void finalSoftmax(float* A, float* B, float sumV){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= 4096){
        return;
    }
    B[i] = A[i]/(sumV);
}


__global__ void checkCMate(float* board1, float* movVec, int pI, int pK, int bKx, int bKy, float repV, float* board){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    //__shared__ float board[64*sizeof(float)];
    board[(x + y*8)+((pI + pK*64)*64)]  = board1[x + y*8];
    board += ((pI + pK*64)*64);
    __syncthreads();
    if(x == 0 && y == 0){
        if(pK >= 7*8){
            if((int)board[pI] == 1){
                board[pK] = 5.0;
                board[pI] = 0.0;
                goto skipA32;
            }
        }
        board[pK] = board[pI];
        board[pI] = 0;
        skipA32:
    }


    //printf("Started checking\n");

    movVec += pI + pK*64;
    int fullShift = x + y*8;
    __syncthreads();
    int pType = -(int)board[fullShift];
    if(pType <= 0){
        return;
    }
    int wKingPos = bKx + bKy*8;
    int sDx = bKx-x;
    int sDy = bKy-y;
    int dx = abs(bKx-x);
    int dy = abs(bKy-y);
    int loopDeltaX = 1;
    int loopDeltaY = 1;
    if(sDx < 0){
        loopDeltaX = -1;
    }
    if(sDy < 0){
        loopDeltaY = -1;
    }

    switch(pType){
        case 1:
            if((bKy == y-1) && dx == 1){
                atomicExch(movVec, repV);
            }
            return;
        case 2:
            if((dx == 1 && dy == 2) || (dx == 2 && dy == 1)){
                atomicExch(movVec, repV);
            }
            break;
        case 3:
            if(dx == dy){
                int ty = y+loopDeltaY;
                for(int tx = x+loopDeltaX; tx != bKx; tx += loopDeltaX){
                    if((int)board[tx + 8*ty] != 0){
                        goto skipLoop3;
                    }
                    ty += loopDeltaY;
                }
                atomicExch(movVec, repV);
            }
            skipLoop3:
            break;
        case 4:
            if(x == bKx){
                for(int ty = y+loopDeltaY; ty != bKy; ty += loopDeltaY){
                    if((int)board[x + 8*ty] != 0){
                        goto skipLoop4;
                    }
                }
                atomicExch(movVec, repV);
            }
            else if(y == bKy){
                for(int tx = x+loopDeltaX; tx != bKx; tx += loopDeltaX){
                    if((int)board[tx + 8*y] != 0){
                        goto skipLoop4;
                    }
                }
                atomicExch(movVec, repV);
            }
            skipLoop4:
            break;
        case 5:
            if(dx == dy){
                int ty = y+loopDeltaY;
                for(int tx = x+loopDeltaX; tx != bKx; tx += loopDeltaX){
                    if((int)board[tx + 8*ty] != 0){
                        goto skipLoop5;
                    }
                    ty += loopDeltaY;
                }
                atomicExch(movVec, repV);
            }
            else if(x == bKx){
                for(int ty = y+loopDeltaY; ty != bKy; ty += loopDeltaY){
                    if((int)board[x + 8*ty] != 0){
                        goto skipLoop5;
                    }
                }
                atomicExch(movVec, repV);
            }
            else if(y == bKy){
                for(int tx = x+loopDeltaX; tx != bKx; tx += loopDeltaX){
                    if((int)board[tx + 8*y] != 0){
                        goto skipLoop5;
                    }
                }
                atomicExch(movVec, repV);
            }
            skipLoop5:
            break;
        case 6:
            if((dx <= 1 && dy <= 1)){
                atomicExch(movVec, repV);
            }
            break;
        default:
            return;
    }
}

__global__ void checkLegalMoves(float* board, float* movVec, float repV, int iSize, int kSize, int xKingP, int yKingP, float* buffer){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    if(i >= iSize || k >= kSize){
        return;
    }

    int pType = (int)(board[i]);
    int oType = (int)(board[k]);
    if(oType > 0){
        movVec[i + k*iSize] = repV;
        return;
    }
    int x0 = i%8;
    int y0 = i/8;
    int x1 = k%8;
    int y1 = k/8;
    int Adx = abs(x1-x0);
    int Ady = abs(y1-y0);
    int lAddX = -1;
    int lAddY = -1;
    if(x1 > x0){
        lAddX = 1;
    }
    if(y1 > y0){
        lAddY = 1;
    }
    switch(pType){
        case 1:
            if(k-i == 8 && oType == 0){
                break;
            }
            if((y1-y0 == 1 && Adx == 1) && oType < 0){
                break;
            }
            if(x0 == x1 && y1 == 3 && y0 == 1 && oType == 0){
                if((int)board[k-8] == 0){
                    break;
                }
            }
            goto skipAll1;
        case 2:
            if((Adx == 1 && Ady == 2) || (Ady == 1 && Adx == 2)){
                break;
            }
            goto skipAll1;
        case 3:
            if(Adx == Ady){
                int ty = y0+lAddY;
                for(int tx = x0+lAddX; tx != x1; tx += lAddX){
                    if((int)board[tx + ty*8] != 0){
                        goto skipAll1;
                    }
                    ty += lAddY;
                }
                break;
            }
            goto skipAll1;
        case 4:
            if(Adx == 0){
                for(int ty = y0+lAddY; ty != y1; ty += lAddY){
                    if((int)board[x0 + ty*8] != 0){
                        goto skipAll1;
                    }
                }
                break;
            }
            if(Ady == 0){
                for(int tx = x0+lAddX; tx != x1; tx += lAddX){
                    if((int)board[tx + y0*8] != 0){
                        goto skipAll1;
                    }
                }
                break;
            }
            goto skipAll1;
        
        case 5:
            if(Adx == Ady){
                int ty = y0+lAddY;
                for(int tx = x0+lAddX; tx != x1; tx += lAddX){
                    if((int)board[tx + ty*8] != 0){
                        goto skipAll1;
                    }
                    ty += lAddY;
                }
                break;
            }
            if(Adx == 0){
                for(int ty = y0+lAddY; ty != y1; ty += lAddY){
                    if((int)board[x0 + ty*8] != 0){
                        goto skipAll1;
                    }
                }
                break;
            }
            if(Ady == 0){
                for(int tx = x0+lAddX; tx != x1; tx += lAddX){
                    if((int)board[tx + y0*8] != 0){
                        goto skipAll1;
                    }
                }
                break;
            }
            goto skipAll1;
        case 6:
            if((Adx <= 1 && Ady <= 1)){
                xKingP = x1;
                yKingP = y1;
                break;
            }
        default:
            skipAll1:
            movVec[i + k*iSize] = repV;
            return;
    }

    checkCMate<<<dim3(1, 1), dim3(8, 8)>>>(board, movVec, i, k, xKingP, yKingP, repV, buffer);

}

__global__ void applyAdam(float* grVector, float* varianceV, float* meanV, int size, int iter){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    meanV[i] = beta1*meanV[i] + (1.0-beta1)*grVector[i];
    varianceV[i] = beta2*varianceV[i] + (1.0-beta2)*(grVector[i]*grVector[i]);
    float lm = (meanV[i])/(1.0-pow(beta1, iter));
    float lv = (varianceV[i])/(1.0-pow(beta2, iter));
    grVector[i] = (lm)/(sqrt(lv)+epsilon);
}

void cpuSoftMax(float* vector, int size){
    float m = -FLT_MAX;
    for(int i = 0; i < size; i++){
        if(vector[i] > m){
            m = vector[i];
        }
    }

    expSoftMax<<<1, size>>>(vector, vector, m);
    cudaDeviceSynchronize();
    float s = 0;
    for(int i = 0; i < size; i++){
        s += vector[i];
    }
    finalSoftmax<<<1, size>>>(vector, vector, s);
    cudaDeviceSynchronize();
}

void genRnd(float* vector, int size, float lB, float uB){
    float* hS = (float*)malloc(size*sizeof(float));
    random_device rd3;
    mt19937 gen3(rd3());
    uniform_real_distribution<> dis3(lB, uB);
    for(int i = 0; i < size; i++){
        hS[i] = dis3(gen3);
    }
    cudaMemcpy(vector, hS, sizeof(float)*size, cudaMemcpyHostToDevice);
    free(hS);
}

void flipBoardAndMult(float *board, int *wKingP, int *bKingP) {
    int wk = *wKingP;
    int bk = *bKingP;
    int new_wk = wk, new_bk = bk;

    for (int idx = 0; idx < 64; ++idx) {
        int mirror = 63 - idx;
        if (idx < mirror) {
            float tmp = -board[idx];
            board[idx] = -board[mirror];
            board[mirror] = tmp;
        }
        if (idx == wk)  new_wk = mirror;
        if (idx == bk)  new_bk = mirror;
    }

    *wKingP = new_bk;
    *bKingP = new_wk;
}

void runFullNet(float* NN, float* writeBack, int* FNN, int fnnS){
    for(int i = 0; i < fnnS-1; i++){
        int blocks = ((FNN[i+1]-1)/1024)+1;
        float* bPass = NN+(FNN[i]*FNN[i+1]);
        MatVecMultFusedAddAct<<<blocks, 1024>>>(NN, writeBack, writeBack+FNN[i], bPass, FNN[i+1], FNN[i]);
        NN = bPass + FNN[i+1];
        writeBack += FNN[i];
        cudaDeviceSynchronize();
    }
}

void calculateGradient(float* NN, float* neuralOut, float* writeBack, float* thetaBuffer, float* expOut, int* FNN, int fnnS, int sumF, int neuralSize){
    float* derH;
    cudaMalloc(&derH, FNN[fnnS-1]*sizeof(float));
    dim3 gridSize1(1, ((FNN[fnnS-2])/4)+1);
    dim3 blockSize1(FNN[fnnS-1], 4);
    int relativeNeuralOffset = neuralSize - (FNN[fnnS-1]*(1+FNN[fnnS-2]));
    int relativeOutputOffset = sumF - (FNN[fnnS-1]+FNN[fnnS-2]);
    finalLayerGradient<<<gridSize1, blockSize1>>>(
        NN+relativeNeuralOffset, neuralOut+relativeOutputOffset, expOut, writeBack+relativeNeuralOffset, FNN[fnnS-1], FNN[fnnS-2], derH);
    cudaDeviceSynchronize();
    computeLastTheta<<<gridSize1, blockSize1>>>(NN+relativeNeuralOffset, derH, thetaBuffer, FNN[fnnS-1], FNN[fnnS-2]);
    cudaDeviceSynchronize();
    sumReduction<<<gridSize1, blockSize1>>>(thetaBuffer, thetaBuffer, FNN[fnnS-1], FNN[fnnS-2], 1);
    cudaDeviceSynchronize();
    for(int i = fnnS-2; i > 0; i--){
        relativeNeuralOffset -= FNN[i]*(1+FNN[i-1]);
        relativeOutputOffset -= FNN[i-1];
        dim3 locGridSize(((FNN[i]-1)/4)+1, ((FNN[i-1]-1)/4)+1);
        dim3 locBlockSize(4, 4);
        dim3 lastLGridSize(1, ((FNN[i-1]-1)/4)+1);
        dim3 lastLBlockSize(4, 4); 
        layerTolayerGradient<<<locGridSize, locBlockSize>>>(
            NN+relativeNeuralOffset, neuralOut+relativeOutputOffset, thetaBuffer, thetaBuffer+(FNN[i]*FNN[i-1]), writeBack+relativeNeuralOffset, FNN[i], FNN[i-1]);
        cudaDeviceSynchronize();
        sumReduction<<<locGridSize, locBlockSize>>>(thetaBuffer+(FNN[i]*FNN[i-1]), thetaBuffer, FNN[i], FNN[i-1], 4);//there's a possibility that part of the buffer is
        cudaDeviceSynchronize();                                                                             //overwritten before it can be added. Fix that
        sumReduction<<<lastLGridSize, lastLBlockSize>>>(thetaBuffer, thetaBuffer, 4, FNN[i-1], 1);
        cudaDeviceSynchronize();
    }
    cudaFree(derH);
}

struct bHash{
    uint64_t* boards;
    uint8_t* rPosSelect;
    uint8_t* rDestSelect;
    int cSize;
    int fSize;
};

void hashBoard(float* cBoard, struct bHash* writeBack, uint8_t pS, uint8_t pD){
    if(writeBack->fSize <= writeBack->cSize+4){
        writeBack->fSize *= 2;
        writeBack->boards = (uint64_t*)realloc(writeBack->boards, writeBack->fSize*8);
        writeBack->rPosSelect = (uint8_t*)realloc(writeBack->rPosSelect, writeBack->fSize);
        writeBack->rDestSelect = (uint8_t*)realloc(writeBack->rDestSelect, writeBack->fSize);
    }

    int relPos = (writeBack->cSize)/4;
    writeBack->rPosSelect[relPos] = pS;
    writeBack->rDestSelect[relPos] = pD;

    uint64_t* lB = &writeBack->boards[writeBack->cSize];
    writeBack->cSize += 4;
    for(int i = 0; i < 4; i++){
        uint64_t wrt = 0;
        for(int k = 0; k < 16; k++){
            int64_t piV = (int64_t)cBoard[k + i*16];
            wrt |= (piV<<(k*4));
        }
        lB[i] = wrt;
    }
}

uint16_t searchHash(float* cBoard, struct bHash* table){
    if(table->cSize <= 0){
        return 0;
    }

    uint64_t lookUp[4];
    for(int i = 0; i < 4; i++){
        uint64_t wrt = 0;
        for(int k = 0; k < 16; k++){
            int64_t piV = (int64_t)cBoard[k + i*16];
            wrt |= (piV<<(k*4));
        }
        lookUp[i] = wrt;
    }

    int searchSpace = (table->cSize)/4;

    for(int i = 0; i < searchSpace; i++){
        int rP = i*4;
        if(table->boards[rP] == lookUp[0] && table->boards[rP+1] == lookUp[1] && table->boards[rP+2] == lookUp[2] && table->boards[rP+3] == lookUp[3]){
            return ((uint16_t)table->rPosSelect[i])|(((uint16_t)table->rDestSelect[i])<<8);
        }
    }
    return 0;
}

float runSelfPlay(float* NN, float* startingBoard, int* FNN, int fnnS, int neuralSize, int sumF, int maxLV){
    struct bHash bRec;
    bRec.boards = NULL;
    bRec.rPosSelect = NULL;
    bRec.rDestSelect = NULL;
    bRec.cSize = 0;
    bRec.fSize = 4;

    //beggining managed memory declarations
    float* realBoard;
    float* probDist;
    float* predictionVector;
    cudaMallocManaged(&realBoard, 64*sizeof(float));
    cudaMallocManaged(&probDist, 4096*sizeof(float));
    cudaMallocManaged(&predictionVector, 128*sizeof(float));
    for(int i = 0; i < 64; i++){
        realBoard[i] = startingBoard[i];
    }
    for(int i = 0; i < 4096; i++){
        probDist[i] = 0.0f;
    }
    for(int i = 0; i < 128; i++){
        predictionVector[i] = 0.0f;
    }
    //end of MM declarations and initialization

    //beggining of device memory declaration and initialization
    float* devBaseNN;
    float* devModNN;
    float* sgdAccumulator;
    float* sgdPointer;
    float* thetaHolder;
    float* movCheckBuffer;
    float* outputVector;
    cudaMalloc(&devBaseNN, neuralSize*sizeof(float));
    cudaMalloc(&devModNN, neuralSize*sizeof(float));
    cudaMalloc(&sgdAccumulator, neuralSize*sizeof(float));
    cudaMalloc(&sgdPointer, neuralSize*sizeof(float));
    cudaMalloc(&thetaHolder, maxLV*maxLV*2*sizeof(float));
    cudaMalloc(&movCheckBuffer, 4096*64*sizeof(float));
    cudaMalloc(&outputVector, sumF*sizeof(float));

    initiateToC<<<neuralSize, 1>>>(sgdAccumulator, 0, neuralSize);
    cudaMemcpy(devBaseNN, NN, neuralSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devModNN, NN, neuralSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    //end of declaration of device side memory

    random_device rnDev;
    mt19937 rng(rnDev());

    for(int g = 0; g < totGames; g++){       
        int wKingPos = 4;
        int bKingPos = 4 + 7*8;
        int tMovR = 0;

        for(int m = 0; m < maxMoves; m++){
            char sk32 = 0;
            uint16_t strP;
            float lMax2;
            float sftmSum;
            uniform_real_distribution<> dist1(0.0, 1.0);
            float rndV4;
            int mSource;
            int mDest;

            if(m == 0 && g%2 == 1){
                goto bgBlack;
            }
            tMovR++;
            //Move prediction setup     ------------------------------------------------
            strP = searchHash(realBoard, &bRec);

            if(strP){
                int16_t srcP = (strP<<8)>>8;
                int16_t dstP = strP>>8;
                for(int i = 0; i < 64; i++){
                    predictionVector[i] = 0;
                    predictionVector[i+64] = 0;
                    if(i == dstP){
                        predictionVector[i+64] = 1.0;
                    }
                    if(i == srcP){
                        predictionVector[i] = 1.0;
                    }
                }
            }
            else{
                uniform_real_distribution<> dist0(0.0, 1.0);
                for(int i = 0; i < 4096; i++){
                    if(i%64 == i/64){
                        probDist[i] = -1.0;
                        continue;
                    }
                    probDist[i] = dist0(rng);
                }
                checkLegalMoves<<<dim3(4, 4), dim3(16, 16)>>>(realBoard, probDist, -FLT_MAX, 64, 64, wKingPos%8, wKingPos/8, movCheckBuffer);
                cudaDeviceSynchronize();
                float clMax = -FLT_MAX;
                int srcI = 0;
                int dstI = 0;
                for(int i = 0; i < 4096; i++){
                    if(probDist[i] > clMax){
                        srcI = i%64;
                        dstI = i/64;
                        clMax = probDist[i];
                    }
                }
                hashBoard(realBoard, &bRec, (uint8_t)srcI, (uint8_t)dstI);
                for(int i = 0; i < 64; i++){
                    predictionVector[i] = 0.0;
                    predictionVector[i+64] = 0.0;
                    if(i == srcI){
                        predictionVector[i] = 1.0;
                    }
                    if(i == dstI){
                        predictionVector[i+64] = 1.0;
                    }
                }
            }
            //end of move prediction setup          ------------------------------------------

            cudaMemcpy(outputVector, realBoard, 64*sizeof(float), cudaMemcpyDefault);
            runFullNet(devModNN, outputVector, FNN, fnnS);
            calculateGradient(devModNN, outputVector, sgdPointer, thetaHolder, predictionVector, FNN, fnnS, sumF, neuralSize);
            //if(strP){
            //    VecCMult<<<((neuralSize-1)/1024)+1, 1024>>>(sgdPointer, sgdPointer, 0.05, neuralSize);
            //}
            VecAdd<<<((neuralSize-1)/1024)+1, 1024>>>(sgdAccumulator, sgdPointer, sgdAccumulator, neuralSize);

            loopBack3:

            generateDistribution<<<dim3(2, 2), dim3(32, 32)>>>(outputVector+sumF-128, probDist);
            cudaDeviceSynchronize();
            checkLegalMoves<<<dim3(4, 4), dim3(16, 16)>>>(realBoard, probDist, -FLT_MAX, 64, 64, wKingPos%8, wKingPos/8, movCheckBuffer);
            cudaDeviceSynchronize();

            for(int i = 0; i < 4096; i++){
                if(probDist[i] > -FLT_MAX){
                    if(i%64 == i/64){
                        continue;
                    }
                    goto skipWLoss;
                }
            }
            break;
            skipWLoss:

            lMax2 = -FLT_MAX;
            for(int i = 0; i < 4096; i++){
                if(probDist[i] > lMax2){
                    lMax2 = probDist[i];
                }
            }
            expSoftMax<<<4, 1024>>>(probDist, probDist, lMax2);
            cudaDeviceSynchronize();
            sftmSum = 0;
            for(int i = 0; i < 4096; i++){
                sftmSum += probDist[i];
            }
            finalSoftmax<<<4, 1024>>>(probDist, probDist, sftmSum);
            rndV4 = dist1(rng);
            mSource = 0;
            mDest = 0;
            cudaDeviceSynchronize();
            for(int i = 0; i < 4096; i++){
                rndV4 -= probDist[i];
                if(rndV4 <= 0.0){
                    mSource = i%64;
                    mDest = i/64;
                    break;
                }
            }
            if(mSource == mDest){
                break;
            }
            if(mSource == wKingPos){
                wKingPos = mDest;
            }
            if(mDest >= 7*8 && (int)realBoard[mSource] == 1){
                realBoard[mDest] = 5.0;
            }
            else{
                realBoard[mDest] = realBoard[mSource];
            }
            realBoard[mSource] = 0.0;
            flipBoardAndMult(realBoard, &wKingPos, &bKingPos);
            if(!sk32){
                bgBlack:
                sk32 = 1;
                cudaMemcpy(outputVector, realBoard, 64*sizeof(float), cudaMemcpyDefault);
                runFullNet(devBaseNN, outputVector, FNN, fnnS);
                goto loopBack3;
            }
        }
        VecCMult<<<((neuralSize-1)/1024)+1, 1024>>>(sgdAccumulator, sgdAccumulator, (1.0/tMovR)*learningStep, neuralSize);  //use adam later on
        cudaDeviceSynchronize();
        VecAdd<<<((neuralSize-1)/1024)+1, 1024>>>(devModNN, sgdAccumulator, devModNN, neuralSize);
        cudaDeviceSynchronize();
        initiateToC<<<neuralSize, 1>>>(sgdAccumulator, 0, neuralSize);
        cudaDeviceSynchronize();
        for(int i = 0; i < 64; i++){
            realBoard[i] = startingBoard[i];
        }
        cudaDeviceSynchronize();
        printf("finished game %d; Size of list is cur: %d\n", g, bRec.cSize);
    }

    float costAcc = 0.0;
    for(int vG = 0; vG < verificationGames; vG++){
        int wKingPos = 4;
        int bKingPos = 4 + 7*8;
        for(int m = 0; m < maxMoves; m++){
            float cAdd = -1.0;
            char sk32 = 0;
            char draw;
            float lMaxV2;
            float sftS;
            uniform_real_distribution<> distV1(0.0, 1.0);
            float rndValid;
            int mSource;
            int mDest;

            if(m == 0 && vG%2 == 1){
                goto valBGBlack;
            }

            cudaMemcpy(outputVector, realBoard, 64*sizeof(float), cudaMemcpyDefault);
            runFullNet(devModNN, outputVector, FNN, fnnS);

            loopBackValidation:

            generateDistribution<<<dim3(2, 2), dim3(32, 32)>>>(outputVector+sumF-128, probDist);
            cudaDeviceSynchronize();
            checkLegalMoves<<<dim3(4, 4), dim3(16, 16)>>>(realBoard, probDist, -FLT_MAX, 64, 64, wKingPos%8, wKingPos/8, movCheckBuffer);
            cudaDeviceSynchronize();
 
            draw = 0;
            for(int i = 0; i < 4096; i++){
                if(probDist[i] > -FLT_MAX){
                    if(i%64 == i/64){
                        draw = 1;
                        continue;
                    }
                    goto skipWLVal;
                }
            }
            if(draw){
                break;
            }
            costAcc += cAdd;
            break;
            skipWLVal:

            lMaxV2 = -FLT_MAX;
            for(int i = 0; i < 4096; i++){
                if(probDist[i] > lMaxV2){
                    lMaxV2 = probDist[i];
                }
            }
            expSoftMax<<<4, 1024>>>(probDist, probDist, lMaxV2);
            cudaDeviceSynchronize();
            sftS = 0.0;
            for(int i = 0; i < 4096; i++){
                sftS += probDist[i];
            }
            finalSoftmax<<<4, 1024>>>(probDist, probDist, sftS);
            rndValid = distV1(rng);
            mSource = 0;
            mDest = 0;
            cudaDeviceSynchronize();
            for(int i = 0; i < 4096; i++){
                rndValid -= probDist[i];
                if(rndValid <= 0.0){
                    mSource = i%64;
                    mDest = i/64;
                    break;
                }
            }
            if(mSource == mDest){
                costAcc += cAdd;
                break;
            }
            if(mSource == wKingPos){
                wKingPos = mDest;
            }
            if(mDest >= 7*8 && (int)realBoard[mSource] == 1){
                realBoard[mDest] = 5.0;
            }
            else{
                realBoard[mDest] = realBoard[mSource];
            }
            realBoard[mSource] = 0.0;
            flipBoardAndMult(realBoard, &wKingPos, &bKingPos);

            if(!sk32){
                valBGBlack:
                sk32 = 1;
                cAdd = 1.0;
                cudaMemcpy(outputVector, realBoard, 64*sizeof(float), cudaMemcpyDefault);
                runFullNet(devBaseNN, outputVector, FNN, fnnS);
                goto loopBackValidation;
            }
        }
        cudaDeviceSynchronize();
        for(int i = 0; i < 64; i++){
            realBoard[i] = startingBoard[i];
        }
    }
    costAcc /= verificationGames;
    if(costAcc > 0.001){
        cudaMemcpy(NN, devModNN, neuralSize*sizeof(float), cudaMemcpyDeviceToHost);
    }
    free(bRec.boards);
    free(bRec.rPosSelect);
    free(bRec.rDestSelect);

    cudaFree(realBoard);
    cudaFree(probDist);
    cudaFree(predictionVector);

    cudaFree(devBaseNN);
    cudaFree(devModNN);
    cudaFree(sgdAccumulator);
    cudaFree(sgdPointer);
    cudaFree(thetaHolder);
    cudaFree(movCheckBuffer);
    cudaFree(outputVector);

    return costAcc;
}

void setupNN(float* gvBoard, float* NN, int* FNN, int fnnS){
    int maxLL = FNN[0];
    int netSum = FNN[0];
    int nSize = 0;
    for(int i = 1; i < fnnS; i++){
        nSize += (FNN[i-1]+1)*FNN[i];
        netSum += FNN[i];
        if(FNN[i] > maxLL){
            maxLL = FNN[i];
        }
    }
    float acm = 0.0;
    while(acm < costTH){
        float pc = runSelfPlay(NN, gvBoard, FNN, fnnS, nSize, netSum, maxLL);
        printf("Completed one validation, cost was: %f\n", pc);
        if(pc > 0){
            acm += pc;
        }
        writeF(NN, nSize, sizeof(float), FNNpath);
        writeF(NN, nSize, sizeof(float), backupPath);
    }
}

void runInputRun(float* gvBoard, float* NN, int* FNN, int fnnS){
    int neuralSize = FNN[0];
    int sumF = 0;
    for(int i = 1; i < fnnS; i++){
        neuralSize += (FNN[i-1]+1)*FNN[i];
        sumF += FNN[i];
    }

    //beggining managed memory declarations
    float* realBoard;
    float* probDist;
    cudaMallocManaged(&realBoard, 64*sizeof(float));
    cudaMallocManaged(&probDist, 4096*sizeof(float));
    for(int i = 0; i < 64; i++){
        realBoard[i] = gvBoard[i];
    }
    for(int i = 0; i < 4096; i++){
        probDist[i] = 0.0f;
    }
    //end of MM declarations and initialization
    
    //beggining of device memory declaration and initialization
    float* devBaseNN;
    float* movCheckBuffer;
    float* outputVector;
    cudaMalloc(&devBaseNN, neuralSize*sizeof(float));
    cudaMalloc(&movCheckBuffer, 4096*64*sizeof(float));
    cudaMalloc(&outputVector, sumF*sizeof(float));

    cudaMemcpy(devBaseNN, NN, neuralSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    //end of declaration of device side memory

    int wKingPos = 4;
    int bKingPos = 4 + 7*8;
    while(1){
        cudaMemcpy(outputVector, realBoard, 64*sizeof(float), cudaMemcpyDefault);
        runFullNet(devBaseNN, outputVector, FNN, fnnS);
        generateDistribution<<<dim3(2, 2), dim3(32, 32)>>>(outputVector+sumF-128, probDist);
        cudaDeviceSynchronize();
        checkLegalMoves<<<dim3(4, 4), dim3(16, 16)>>>(realBoard, probDist, -FLT_MAX, 64, 64, wKingPos%8, wKingPos/8, movCheckBuffer);
        cudaDeviceSynchronize();

        char draw = 0;
        for(int i = 0; i < 4096; i++){
            if(probDist[i] > -FLT_MAX){
                if(i%64 == i/64){
                    draw = 1;
                    continue;
                }
                goto skipWLVal;
            }
        }
        if(draw){
            printf("Game ended in a draw.\n");
            break;
        }
        printf("The network lost the game\n");
        break;
        skipWLVal:
        int mSource = 0;
        int mDest = 0;
        float lMmax = -FLT_MAX;
        for(int i = 0; i < 4096; i++){
            if(probDist[i] > -FLT_MAX){
                printf("TestS: x=%d y=%d|TestD: x=%d y=%d|||Dist was: %f\n", (i%64)%8, (i%64)/8, (i/64)%8, (i/64)/8, probDist[i]);
            }
            if(probDist[i] > lMmax){
                lMmax = probDist[i];
                mSource = i%64;
                mDest = i/64;
            }
        }
        if(mSource == mDest){
            printf("The network resigned.\n");
            break;
        }
        if(mSource == wKingPos){
            wKingPos = mDest;
        }
        if(mDest >= 7*8 && (int)realBoard[mSource] == 1){
            realBoard[mDest] = 5.0;
        }
        else{
            realBoard[mDest] = realBoard[mSource];
        }
        realBoard[mSource] = 0.0;
        printf("Source: x=%d y=%d\nDestination: x=%d y=%d\n", mSource%8, mSource/8, mDest%8, mDest/8);

        int mox;
        int moy;
        int mix;
        int miy;
        float tt = 1;
        printf("Pick move\n");
        scanf("%d|%d %d|%d %f", &mox, &moy, &mix, &miy, &tt);
        int mo = mox+(moy*8);
        int mi = mix+(miy*8);
        //for(int k0 = 0; k0 < 8; k0++){
        //    for(int k1 = 0; k1 < 8; k1++){
        //        printf("%f | ", realBoard[k1 + k0*8]);
        //    }
        //    printf("\n-----------------------------------\n");
        //}
        printf("\n");
        if(mo == bKingPos){
            bKingPos = mi;
        }
        realBoard[mi] = realBoard[mo];
        realBoard[mo] = 0.0;
        if(tt < 0.0){
            realBoard[mi] = tt;
        }

    }


}

float* randomWeihgtHeGeneration(int* layerVals, int totalLayers, int* retSize){
    int fullsize = 0;
    for(int i = 1; i < totalLayers; i++){
        fullsize += (layerVals[i-1] + 1)*layerVals[i];
    }
    *retSize = fullsize;
    float* retVal = (float*)malloc(fullsize*sizeof(float));
    float* movR = retVal;
    random_device rd1;
    mt19937 gen1(rd1());
    for(int i = 1; i < totalLayers; i++){
        normal_distribution<> dis1(0, sqrt(2.0/(layerVals[i-1] + layerVals[i])));
        for(int k = 0; k < layerVals[i-1]*layerVals[i]; k++){
            movR[k] = dis1(gen1);
        }
        for(int k = layerVals[i-1]*layerVals[i]; k < (layerVals[i-1]+1)*layerVals[i]; k++){
            movR[k] = 0;
        }
        movR += (layerVals[i-1] + 1)*layerVals[i];
    }
    return retVal;
}


int main(int argc, char** argv){
    int FNN[] = {256, 1024, 2048, 2048, 4096, 4096, 2048, 2048, 1024, 512, 256, 128};
    float chessBoard[] = {
        4.0, 2.0, 3.0, 5.0, 6.0, 3.0, 2.0, 4.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -4.0, -2.0, -3.0, -5.0, -6.0, -3.0, -2.0, -4.0
    };
    
    //int rSize;
    //float* fnnGen = randomWeihgtHeGeneration(FNN, sizeof(FNN)/sizeof(int), &rSize);
    //writeF(fnnGen, rSize, sizeof(float), FNNpath);
    
    long dmv;
    float* network = (float*)readF(FNNpath, sizeof(float), &dmv);
    runInputRun(chessBoard, network, FNN, sizeof(FNN)/sizeof(int));
    //setupNN(chessBoard, network, FNN, sizeof(FNN)/sizeof(int));
}