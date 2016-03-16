#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <endian.h>
#include <omp.h>

#define calcIndex(width, x,y)  ((y)*(width) + (x))

void show(unsigned* currentfield, int w, int h) {
    printf("\033[H");
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) printf(currentfield[calcIndex(w, x,y)] ? "\033[07m  \033[m" : "  ");
        printf("\033[E");
    }
    printf("\n");
    fflush(stdout);
}

float convert2BigEndian( const float inFloat )
{
    float retVal;
    char *floatToConvert = ( char* ) & inFloat;
    char *returnFloat    = ( char* ) & retVal;

    // swap the bytes into a temporary buffer
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];

    return retVal;
}

void writeVTKParallel(unsigned* currentfield, int w, int h, int t, char* prefix) {
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); i = i + 1)
    {
        int beginHeight = (i % 2) ? h / 2 : 0; // oder ((i % 2) + 1) / 2) * h
        int endHeight = (i % 2) ? h : h / 2; // oder ((i % 2) + 1) / 2) * h
        int beginWidth =  ((double)((i-1)%(omp_get_max_threads()/2) + 1)) / (omp_get_max_threads()/2) * w;
        beginWidth = beginWidth == w ? 0 : beginWidth - 1;
        beginWidth = beginWidth == 0 ? 0 - 1 : beginWidth;
        //beginWidth = beginWidth == -1 ? 0 : beginWidth;

        int endWidth =  ((double)(i%(omp_get_max_threads()/2) + 1)) / (omp_get_max_threads()/2) * w;
        endHeight = endHeight == h ? h : endHeight + 1;

        char name[1024] = "\0";
        char* folder = "out/";
        sprintf(name, "%s%s_%d_%d.vtk",folder, prefix, omp_get_thread_num(), t);
        FILE* outfile = fopen(name, "w");

        //Write vtk header
        fprintf(outfile,"# vtk DataFile Version 3.0\n");
        fprintf(outfile,"frame %d\n", t);
        fprintf(outfile,"BINARY\n");
        fprintf(outfile,"DATASET STRUCTURED_POINTS\n");
        fprintf(outfile,"DIMENSIONS %d %d %d \n", endWidth - beginWidth, endHeight - beginHeight, 1); //local values
        fprintf(outfile,"SPACING 1.0 1.0 1.0\n");//or ASPECT_RATIO
        fprintf(outfile,"ORIGIN %d %d 0\n", beginWidth, beginHeight); //each thread block (w, h)
        fprintf(outfile,"POINT_DATA %d\n", (endWidth - beginWidth) * (endHeight - beginHeight)); //local value
        fprintf(outfile,"SCALARS data float 1\n");
        fprintf(outfile,"LOOKUP_TABLE default\n");

        for (int y = beginHeight; y < endHeight; y++) {
            for (int x = beginWidth; x <= endWidth; x++) {

                float value = currentfield[calcIndex(w, x, y)]; // != 0.0 ? 1.0:0.0;
                value = convert2BigEndian(value);
                fwrite(&value, 1, sizeof(float), outfile);
            }
        }
        fclose(outfile);
    }
}

void writeVTK(unsigned* currentfield, int w, int h, int t, char* prefix) {
    char name[1024] = "\0";
    char* folder = "out/";
    sprintf(name, "%s%s_%d.vtk",folder, prefix, t);
    FILE* outfile = fopen(name, "w");

    /*Write vtk header */
    fprintf(outfile,"# vtk DataFile Version 3.0\n");
    fprintf(outfile,"frame %d\n", t);
    fprintf(outfile,"BINARY\n");
    fprintf(outfile,"DATASET STRUCTURED_POINTS\n");
    fprintf(outfile,"DIMENSIONS %d %d %d \n", w, h, 1);
    fprintf(outfile,"SPACING 1.0 1.0 1.0\n");//or ASPECT_RATIO
    fprintf(outfile,"ORIGIN 0 0 0\n");
    fprintf(outfile,"POINT_DATA %d\n", h*w);
    fprintf(outfile,"SCALARS data float 1\n");
    fprintf(outfile,"LOOKUP_TABLE default\n");

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float value = currentfield[calcIndex(w, x,y)]; // != 0.0 ? 1.0:0.0;
            value = convert2BigEndian(value);
            fwrite(&value, 1, sizeof(float), outfile);
        }
    }
    fclose(outfile);
}

int getNumberOfAliveCells(unsigned* currentfield, int w, int h, int x, int y) {
    int aliveFields = 0;
    for (int j = y - 1; j <= y + 1; j = j + 1) {
        for (int i = x - 1; i <= x + 1; i = i + 1) {
            if (currentfield[calcIndex(w, i,j)]) {
                aliveFields += 1;
            }
        }
    }
    return aliveFields;
}

int evolve(unsigned* currentfield, unsigned* newfield, int w, int h) {
    int changed = 0;
#pragma omp parallel for reduction(+:changed)
    for (int i = 0; i < omp_get_max_threads(); i = i + 1)
    {
        int beginHeight = (i % 2) ? h / 2 + 1 : 0; // oder ((i % 2) + 1) / 2) * h
        beginHeight = beginHeight == 0 ? 0 + 1 : beginHeight;
        int endHeight = (i % 2) ? h : h / 2; // oder ((i % 2) + 1) / 2) * h
        endHeight = endHeight == h ? h : endHeight + 1;

        int beginWidth =  ((double)((i-1)%(omp_get_max_threads()/2) + 1)) / (omp_get_max_threads()/2) * w;
        beginWidth = beginWidth == w ? 0 + 1 : beginWidth;
        beginWidth = beginWidth == 0 ? 0 + 1 : beginWidth;
        int endWidth =  ((double)(i%(omp_get_max_threads()/2) + 1)) / (omp_get_max_threads()/2) * w;
        //endWidth = endWidth == w ? w : endWidth + 1;
        beginWidth = beginWidth == w ? 0 : beginWidth - 1;
        beginWidth = beginWidth == 0 ? 0 : beginWidth;

        for (int y = beginHeight; y < endHeight - 1; y++) {
            for (int x = beginWidth; x <= endWidth - 1; x++) {

                int numberOfAliveCells = getNumberOfAliveCells(currentfield, w, h, x, y);
                int currentfieldValue = currentfield[calcIndex(w, x,y)];
                //Dead Field and 3 Living Fields -> resurrect Field       //Living Field with 2 or 3 living neighbours stays alive
                newfield[calcIndex(w, x,y)] = ((!currentfieldValue && numberOfAliveCells == 3) || (currentfieldValue && (numberOfAliveCells == 3 || numberOfAliveCells == 4)));
                if (newfield[calcIndex(w, x,y)] != currentfield[calcIndex(w, x,y)])
                {
                    changed = 1;
                }
            }
        }
    }

    //randaustausch
#pragma omp parallel for
    for (int x = 0; x < w; x = x + 1)
    {
        newfield[calcIndex(w, x, 0)] = newfield[calcIndex(w, x, h - 2)];
        newfield[calcIndex(w, x, h - 1)] = newfield[calcIndex(w, x, 1)];
    }
#pragma omp parallel for
    for (int y = 0; y < h; y = y + 1)
    {
        newfield[calcIndex(w, 0, y)] = newfield[calcIndex(w, w - 2, y)];
        newfield[calcIndex(w, w - 1, y)] = newfield[calcIndex(w, 1, y)];
    }
    return changed;
}

void filling(unsigned* currentfield, int w, int h) {
    for (int i = 0; i < h*w; i++) {
        currentfield[i] = (rand() < RAND_MAX / 10) ? 1 : 0; ///< init domain randomly
    }
}

void game(int w, int h, int timesteps) {
    unsigned *currentfield = calloc(w*h, sizeof(unsigned));
    unsigned *newfield     = calloc(w*h, sizeof(unsigned));

    filling(currentfield, w, h);
    for (int t = 0; t < timesteps; t++) {
        //show(currentfield, w, h);
        int changes = 0;
#pragma omp sections
        {
#pragma omp section
            {
                writeVTKParallel(currentfield, w, h, t, "output");
            }
#pragma omp section
            {
                changes = evolve(currentfield, newfield, w, h);
            }
        }
        if (changes == 0) {
            sleep(3);
            break;
        }
        
        //SWAP
        unsigned *temp = currentfield;
        currentfield = newfield;
        newfield = temp;
    }
    
    free(currentfield);
    free(newfield);
}

int main(int c, char **v) {
    
    int w = 0, h = 0, timesteps = 10;
    if (c > 1) w = atoi(v[1]) + 2; ///< read width
    if (c > 2) h = atoi(v[2]) + 2; ///< read height
    if (c > 3) timesteps = atoi(v[3]);
    if (w <= 0) w = 30 + 2; ///< default width
    if (h <= 0) h = 30 + 2; ///< default height
    if (c <= 0) c = 10; ///< default height
    game(w, h, timesteps);
}
