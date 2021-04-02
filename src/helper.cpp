#include "helper.h"  // helper.h must be in the current directory. or use relative or absolute path to it. e.g #include "include/helper.h"

void writeLog(std::string filename, std::string str)
{

    FILE *fp;

    fp = fopen(filename.c_str(), "a");
    if (fp == NULL)
    {
        perror("Error");
        exit(1);
    }
    else
    {

        fprintf(fp, "%s", str.c_str());
    }

    fclose(fp);
}


