#include <iostream>
#include <opencv2/opencv.hpp>
#include "stopWatch.hpp"
using namespace cv;
using namespace std;

int main()
{
    char fileName[] = "car.jpg";
    IplImage *image, *newImage;
    image = cvLoadImage(fileName, CV_LOAD_IMAGE_GRAYSCALE);
    stopWatch timer[1];
    if (!image)
    {
        cout << "找不到檔案!!!" << endl;
    }
    else
    {
        timer[0].start();
        newImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
        cvCanny(image, newImage, 40, 80);
        timer[0].stop();
        cvSaveImage("0.jpg", newImage);
        cout << timer[0].elapsedTime() << endl;
    }
    return 0;
}
