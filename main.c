#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int calculateManhattan(int x1, int y1, int x2, int y2){
    x1++;
    x2++;
    y1++;
    y2++;
    return abs(x1 - x2) + abs(y1 - y2); 
}

int*** createData(int count, int size){
    int ***data = calloc(count, sizeof(int **));
    for (int i = 0; i < count; i++){
        data[i] = calloc(size, sizeof(int *));
        for (int j = 0; j < size; j++){
            data[i][j] = calloc(size, sizeof(int));
        }
    }
    return data;
}

void fillAB(int**** data, int dotCount, int dataCount, int dataSize){
    srand(time(NULL));
    int filledCoordinates[dotCount][2];

    int randomX;
    int randomY;

    for (int i = 0; i < dataCount; i++){
        for (int j = 0; j < dotCount; j++){
            filledCoordinates[j][0] = -1;
            filledCoordinates[j][1] = -1;
        }

        for (int fillCount = 0; fillCount < dotCount; fillCount++){

            int continueFindingRandom = 1;

            while (continueFindingRandom){
                randomX = rand() % dataSize;
                randomY = rand() % dataSize;


                int isFilled = 0;
                for (int j = 0; j < fillCount; j++){
                    if (filledCoordinates[j][0] == randomX && filledCoordinates[j][1] == randomY){
                        isFilled = 1;
                    }
                }

                if (isFilled == 0){
                    (*data)[i][randomX][randomY] = 1;
                        filledCoordinates[fillCount][0] = randomX;
                        filledCoordinates[fillCount][1] = randomY;
                        continueFindingRandom = 0;
                }
            }
        }
    }
}

int** getManhattanData(int*** data, int dataCount, int dataSize, int dotCount){
    int **manhattanData = calloc(dataCount, sizeof(int *));
    for (int i = 0; i < dataCount; i++){
        manhattanData[i] = calloc(2, sizeof(int));
    }

    for (int dataIndex = 0; dataIndex < dataCount; dataIndex++){
        int dots[dotCount][2];
        int dotIndex = 0;

        for (int i = 0; i < dataSize; i++){
            for (int j = 0; j < dataSize; j++){
                if (data[dataIndex][i][j] == 1){
                    dots[dotIndex][0] = i;
                    dots[dotIndex][1] = j;
                    dotIndex++;
                }
            }
        }

        int min = 9999;
        int max = -9999;

        for (int i = 0; i < dotCount; i++){
            for (int j = i + 1; j < dotCount; j++){
                int distance = calculateManhattan(dots[i][0], dots[i][1], dots[j][0], dots[j][1]);
                if (distance < min){
                    min = distance;
                }
                if (distance > max){
                    max = distance;
                }
            }
        }

        manhattanData[dataIndex][0] = min;
        manhattanData[dataIndex][1] = max;
    }
    return manhattanData;
}

void writeABtoCSV(const char *filenameA, const char *filenameB, int*** data, int** manhattanData, int dataCount, int dataSize, int normalisedCountA, int normalisedCountB) {
    FILE *fileA = fopen(filenameA, "w"); 
    FILE *fileB = fopen(filenameB, "w"); 
    
    if (fileA == NULL || fileB == NULL) {
        printf("File error!\n");
        return;
    }

    int counts[dataSize * 2][2];

    for (int i = 0; i < dataSize * 2; i++){
        counts[i][0] = 0;
        counts[i][1] = 0;
    }

    for (int i = 0; i < dataCount; i++) {
        if (counts[manhattanData[i][0] - 1][0] < normalisedCountA){
            for (int j = 0; j < dataSize; j++) {
                for (int z = 0; z < dataSize; z++){
                    fprintf(fileA, "%d,", data[i][j][z]);
                }
            }
            fprintf(fileA, "%d\n", manhattanData[i][0]);
            counts[manhattanData[i][0] - 1][0]++;
        }

        if (counts[manhattanData[i][1] - 1][1] < normalisedCountB){
            for (int j = 0; j < dataSize; j++) {
                for (int z = 0; z < dataSize; z++){
                    fprintf(fileB, "%d,", data[i][j][z]);
                }
            }
            fprintf(fileB, "%d\n", manhattanData[i][1]);
            counts[manhattanData[i][1] - 1][1]++;
        }
    }

    fclose(fileA);
    fclose(fileB);
    printf("Veri basariyla %s dosyasina yazildi", filenameA);
    printf("\nVeri basariyla %s dosyasina yazildi", filenameB);
}

void printCountDisturbationAB(int** manhattanData, int dataCount, int dataSize){
    int counts[dataSize * 2][2];

    for (int i = 0; i < dataSize * 2; i++){
        counts[i][0] = 0;
        counts[i][1] = 0;
    }

    for (int i = 0; i < dataCount; i++){
        counts[manhattanData[i][0] - 1][0]++;
        counts[manhattanData[i][1] - 1][1]++;
    }

    printf("\nMinimum Value Disturbation:");
    for (int i = 0; i < dataSize * 2; i++){
        printf("\nCount for value %d is: %d", i + 1, counts[i][0]);
    }

    printf("\n\nMaximum Value Disturbation:");
        for (int i = 0; i < dataSize * 2; i++){
        printf("\nCount for value %d is: %d", i + 1, counts[i][1]);
    }
}

void ab(int ***data, int dataCount, int dataSize, int ABdotCount){
    fillAB(&data, ABdotCount, dataCount, dataSize);
    int **manhattanData = getManhattanData(data, dataCount, dataSize, ABdotCount);

    for (int i = 0; i < 25; i++){
        for (int j = 0; j < 25; j++){
            printf("%d ", data[703][i][j]);
        }
        printf("\n");
    }
    printf("\n Manhattan Min: %d \n Manhattan Max: %d \n", manhattanData[703][0], manhattanData[703][1]);

    printCountDisturbationAB(manhattanData, dataCount, dataSize);
    writeABtoCSV("a_200.csv", "b_200.csv", data, manhattanData, dataCount, dataSize, 200, 200);
}

int** fillC(int**** data, int maxDotCount, int dataCount, int dataSize){
    srand(time(NULL));
    
    int **cResults = calloc(dataCount, sizeof(int *));
    for (int i = 0; i < dataCount; i++){
        cResults[i] = calloc(3, sizeof(int));
    }

    int randomX;
    int randomY;

    for (int i = 0; i < dataCount; i++){
        int dotCount = (rand() % maxDotCount) + 1; 
        int filledCoordinates[dotCount][2];
        cResults[i][0] = dotCount;
        cResults[i][1] = dotCount % 2;

        for (int j = 0; j < dotCount; j++){
            filledCoordinates[j][0] = -1;
            filledCoordinates[j][1] = -1;
        }

        for (int fillCount = 0; fillCount < dotCount; fillCount++){
            int continueFindingRandom = 1;

            while (continueFindingRandom){
                randomX = rand() % dataSize;
                randomY = rand() % dataSize;

                int isFilled = 0;
                for (int j = 0; j < fillCount; j++){
                    if (filledCoordinates[j][0] == randomX && filledCoordinates[j][1] == randomY){
                        isFilled = 1;
                    }
                }

                if (isFilled == 0){
                    (*data)[i][randomX][randomY] = 1;
                        filledCoordinates[fillCount][0] = randomX;
                        filledCoordinates[fillCount][1] = randomY;
                        continueFindingRandom = 0;
                }
            }
        }
    }
    return cResults;
}

void printCountDisturbationC(int** cResults, int dataCount, int cMaxDotCount, int dataSize){
    int counts[cMaxDotCount][1];
    int parityCounts[2][1];
    int manhattanCounts[dataSize * 2][1];

    for (int i = 0; i < cMaxDotCount; i++){
        counts[i][0] = 0;
    }

    for (int i = 0; i < 2; i++){
        parityCounts[i][0] = 0;
    }

    for (int i = 0; i < dataSize * 2; i++){
        manhattanCounts[i][0] = 0;
    }

    for (int i = 0; i < dataCount; i++){
        counts[cResults[i][0] - 1][0]++;
        parityCounts[cResults[i][1]][0]++;
        manhattanCounts[cResults[i][2] - 1][0]++;
    }

    printf("\nRandom Dot Count Disturbation:");
    for (int i = 0; i < cMaxDotCount; i++){
        printf("\nCount for value %d is: %d", i + 1, counts[i][0]);
    }

    printf("\nParity Count Disturbation:");
    for (int i = 0; i < 2; i++){
        printf("\nCount for value %d is: %d", i, parityCounts[i][0]);
    }

    printf("\nManhattan Distance Disturbation:");
    for (int i = 0; i < dataSize * 2; i++){
        printf("\nCount for value %d is: %d", i + 1, manhattanCounts[i][0]);
    }
}

void writeCDtoCSV(const char *filenameC, const char *filenameD, int*** data, int** cResults, int dataCount, int dataSize, int cMaxDotCount, int normalisedCountC, int normalisedCountD) {
    FILE *fileC = fopen(filenameC, "w"); 
    FILE *fileD = fopen(filenameD, "w"); 
    
    if (fileC == NULL || fileD == NULL) {
        printf("File error!\n");
        return;
    }

    int counts[cMaxDotCount][1];
    int parityCounts[2][1];

    for (int i = 0; i < cMaxDotCount; i++){
        counts[i][0] = 0;
    }
    for (int i = 0; i < 2; i++){
        parityCounts[i][0] = 0;
    }

    for (int i = 0; i < dataCount; i++) {
        if (counts[cResults[i][0] - 1][0] < normalisedCountC){
            for (int j = 0; j < dataSize; j++) {
                for (int z = 0; z < dataSize; z++){
                    fprintf(fileC, "%d,", data[i][j][z]);
                }
            }
            fprintf(fileC, "%d\n", cResults[i][0]);
            counts[cResults[i][0] - 1][0]++;
        }

        if (parityCounts[cResults[i][1]][0] < normalisedCountD){
            for (int j = 0; j < dataSize; j++) {
                for (int z = 0; z < dataSize; z++){
                    fprintf(fileD, "%d,", data[i][j][z]);
                }
            }
            fprintf(fileD, "%d\n", cResults[i][1]);
            parityCounts[cResults[i][1]][0]++;
        }
    }

    fclose(fileC);
    fclose(fileD);
    printf("Veri basariyla %s dosyasina yazildi", filenameC);
    printf("\nVeri basariyla %s dosyasina yazildi", filenameD);
}

void fillE(int*** data, int **cResults, int dataCount, int dataSize){
    for (int i = 0; i < dataCount; i++){
        int dots[cResults[i][0]][2];
        int dotIndex = 0;

        for (int j = 0; j < cResults[i][0]; j++){
            dots[j][0] = 0;
            dots[j][1] = 0;
        }

        for (int j = 0; j < dataSize; j++){
            for (int z = 0; z < dataSize; z++){
                if (data[i][j][z] == 1){
                    dots[dotIndex][0] = j;
                    dots[dotIndex][1] = z;
                    dotIndex++;
                }
            }
        }

        if (cResults[i][1] == 1){
            int min = 9999999;

            for (int j = 0; j < dotIndex; j++){
                int distance = calculateManhattan(dots[j][0], dots[j][1], 0, 0);
                if (distance < min){
                    min = distance;
                }
            }

            cResults[i][2] = min;
        } 
        
        else if (cResults[i][1] == 0){
            int min = 9999999;

            for (int j = 0; j < dotIndex; j++){
                int distance = calculateManhattan(dots[j][0], dots[j][1], 24, 24);
                if (distance < min){
                    min = distance;
                }
            }

            cResults[i][2] = min;
        }
    }
}

void writeEtoCSV(const char *filename, int*** data, int** cResults, int dataCount, int dataSize, int normalisedCount) {
    FILE *file = fopen(filename, "w");  
    
    if (file == NULL) {
        printf("File error!\n");
        return;
    }

    int counts[dataSize * 2][1];

    for (int i = 0; i < dataSize * 2; i++){
        counts[i][0] = 0;
    }

    for (int i = 0; i < dataCount; i++) {
        if (counts[cResults[i][2] - 1][0] < normalisedCount){
            for (int j = 0; j < dataSize; j++) {
                for (int z = 0; z < dataSize; z++){
                    fprintf(file, "%d,", data[i][j][z]);
                }
            }
            fprintf(file, "%d\n", cResults[i][2]);
            counts[cResults[i][2] - 1][0]++;
        }
    }

    fclose(file);
    printf("Veri basariyla %s dosyasina yazildi", filename);
}

void c(int ***data, int dataCount, int dataSize, int cMaxDotCount){
    int** cResults = fillC(&data, cMaxDotCount, dataCount, dataSize);
    fillE(data, cResults, dataCount, dataSize);
    printCountDisturbationC(cResults, dataCount, cMaxDotCount, dataSize);
    writeEtoCSV("e_200.csv", data, cResults, dataCount, dataSize, 200);
    // writeCDtoCSV("c_500.csv", "d_1000.csv", data, cResults, dataCount, dataSize, cMaxDotCount, 500, 1000);
}

void resetData(int ***data, int dataCount, int dataSize){
    for (int i = 0; i < dataCount; i++){
        for (int j = 0; j < dataSize; j++){
            for (int z = 0; z < dataSize; z++){
                data[i][j][z] = 0;
            }
        }
    }
}

int main(){
    int dataCount = 5000000;
    int dataSize = 25;
    int ABdotCount = 5;
    int cMaxDotCount = 10;
    
    int ***data = createData(dataCount, dataSize);
    ab(data, dataCount, dataSize, ABdotCount);
    resetData(data, dataCount, dataSize);
    c(data, dataCount, dataSize, cMaxDotCount);
    
    return 0;
}