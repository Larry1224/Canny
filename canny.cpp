#include <iostream>
#include <opencv2/opencv.hpp>
#include "stopWatch.hpp"
using namespace cv; // 使用命名空間cv

/**
 將兩個圖像拼接，在同一個窗口顯示
 dst 輸出拼接後的圖像
 src1 拼接的第一幅圖
 src2 拼接的第二幅圖
 */
void mergeImg(Mat &dst, Mat &src1, Mat &src2)
{
    int rows = src1.rows;
    int cols = src1.cols + 5 + src2.cols;
    CV_Assert(src1.type() == src2.type());
    dst.create(rows, cols, src1.type());
    src1.copyTo(dst(Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(dst(Rect(src1.cols + 5, 0, src2.cols, src2.rows)));
}

/**
 一维高斯卷積，對每行進行高斯卷積
 img 輸入原圖
 dst  一維高斯卷積後圖像
 */
void gaussianConvolution(Mat &img, Mat &dst)
{
    int nr = img.rows;
    int nc = img.cols;
    int templates[3] = {1, 2, 1};

    // 每行邊緣點的所有點
    for (int j = 0; j < nr; j++)
    {
        uchar *data = img.ptr<uchar>(j); //提取該行地址
        for (int i = 1; i < nc - 1; i++)
        {
            int sum = 0;
            for (int n = 0; n < 3; n++)
            {
                sum += data[i - 1 + n] * templates[n]; //相乘累加
            }
            sum /= 4;
            dst.ptr<uchar>(j)[i] = sum;
        }
    }
}

/**
 高斯濾波器，利用3*3的高斯模版進行高斯卷積
 img 輸入原圖
 dst  高斯濾波後的輸出圖像
*/
void gaussianFilter(Mat &img, Mat &dst)
{
    // 對水平方向進行濾波
    Mat dst1 = img.clone();
    gaussianConvolution(img, dst1);
    //圖像矩陣轉置
    Mat dst2;
    transpose(dst1, dst2);
    // 對垂直方向進行濾波
    Mat dst3 = dst2.clone();
    gaussianConvolution(dst2, dst3);
    // 再次轉置
    transpose(dst3, dst);
}

/**
 用一階偏導有限差分計算梯度幅值和方向
 img 輸入原圖
 gradXY 輸出的梯度幅值
 theta 輸出的梯度方向
 */
void getGrandient(Mat &img, Mat &gradXY, Mat &theta)
{
    gradXY = Mat::zeros(img.size(), CV_8U);
    theta = Mat::zeros(img.size(), CV_8U);

    for (int j = 1; j < img.rows - 1; j++)
    {
        for (int i = 1; i < img.cols - 1; i++)
        {
            // double gradY = double(img.ptr<uchar>(j - 1)[i - 1] + img.ptr<uchar>(j - 1)[i] + img.ptr<uchar>(j - 1)[i + 1] - img.ptr<uchar>(j + 1)[i - 1] - img.ptr<uchar>(j + 1)[i] - img.ptr<uchar>(j + 1)[i + 1]) / 3;
            // double gradX = double(img.ptr<uchar>(j - 1)[i + 1] + img.ptr<uchar>(j)[i + 1] + img.ptr<uchar>(j + 1)[i + 1] - img.ptr<uchar>(j - 1)[i - 1] - img.ptr<uchar>(j)[i - 1] - img.ptr<uchar>(j + 1)[i - 1]) / 3;
            double gradY = (img.ptr<uchar>(j)[i + 1] - img.ptr<uchar>(j)[i] + img.ptr<uchar>(j + 1)[i + 1] - img.ptr<uchar>(j + 1)[i]);
            double gradX = (img.ptr<uchar>(j + 1)[i] - img.ptr<uchar>(j)[i] + img.ptr<uchar>(j + 1)[i + 1] - img.ptr<uchar>(j)[i + 1]);
            gradXY.ptr<uchar>(j)[i] = sqrt(gradX * gradX + gradY * gradY); //計算梯度
            theta.ptr<uchar>(j)[i] = atan(gradY / gradX);                  //計算梯度方向
        }
    }
}

/**
 局部非極大值抑制
 gradXY 輸入的梯度幅值
 theta 輸入的梯度方向
 dst 輸出的經局部非極大值抑制後的圖像
 */
void nonLocalMaxValue(Mat &gradXY, Mat &theta, Mat &dst)
{
    dst = gradXY.clone();
    for (int j = 1; j < gradXY.rows - 1; j++)
    {
        for (int i = 1; i < gradXY.cols - 1; i++)
        {
            double t = double(theta.ptr<uchar>(j)[i]);
            double g = double(dst.ptr<uchar>(j)[i]);
            if (g == 0.0)
            {
                continue; //若已是最小值則不繼續判斷
            }
            double g0, g1;
            if ((t >= -(3 * M_PI / 8)) && (t < -(M_PI / 8)))
            {
                g0 = double(dst.ptr<uchar>(j - 1)[i - 1]);
                g1 = double(dst.ptr<uchar>(j + 1)[i + 1]);
            }
            else if ((t >= -(M_PI / 8)) && (t < M_PI / 8))
            {
                g0 = double(dst.ptr<uchar>(j)[i - 1]);
                g1 = double(dst.ptr<uchar>(j)[i + 1]);
            }
            else if ((t >= M_PI / 8) && (t < 3 * M_PI / 8))
            {
                g0 = double(dst.ptr<uchar>(j - 1)[i + 1]);
                g1 = double(dst.ptr<uchar>(j + 1)[i - 1]);
            }
            else
            {
                g0 = double(dst.ptr<uchar>(j - 1)[i]);
                g1 = double(dst.ptr<uchar>(j + 1)[i]);
            }
            if (g <= g0 || g <= g1)
            {
                dst.ptr<uchar>(j)[i] = 0.0;
            }
        }
    }
}

/**
 弱邊緣點補充連接強邊緣點
 img 弱邊緣點補充連接強邊緣點的輸入和輸出圖像
 */
void doubleThresholdLink(Mat &img)
{
    // 循還找到強邊緣點，把其區域内的弱邊緣點變為強邊緣點
    for (int j = 1; j < img.rows - 2; j++)
    {
        for (int i = 1; i < img.cols - 2; i++)
        {
            // 如果該點是強邊緣點
            if (img.ptr<uchar>(j)[i] == 255)
            {
                //計算該強邊緣點領域
                for (int m = -1; m < 1; m++)
                {
                    for (int n = -1; n < 1; n++)
                    {
                        // 該點為弱邊緣點（不是強邊緣點，也不是被抑制的0點）
                        if (img.ptr<uchar>(j + m)[i + n] != 0 && img.ptr<uchar>(j + m)[i + n] != 255)
                        {
                            img.ptr<uchar>(j + m)[i + n] = 255; //該弱邊緣緣點補充為強邊緣點
                        }
                    }
                }
            }
        }
    }

    for (int j = 0; j < img.rows - 1; j++)
    {
        for (int i = 0; i < img.cols - 1; i++)
        {
            // 如果該點依舊是弱邊緣點，即此點是孤立邊緣點
            if (img.ptr<uchar>(j)[i] != 255)
            {
                img.ptr<uchar>(j)[i] = 0; //該孤立弱邊緣點抑制
            }
        }
    }
}

/**
 用雙閥值算法檢測和連接邊緣
 low 輸入的低閥值
 high 輸入的高閥值
 img 輸入的原圖像
 dst 輸出的用雙閥值算法檢測和連接邊緣後的圖像
 */
void doubleThreshold(double low, double high, Mat &img, Mat &dst)
{
    dst = img.clone();

    // 區分出弱邊緣點和強邊緣點
    for (int j = 0; j < img.rows - 1; j++)
    {
        for (int i = 0; i < img.cols - 1; i++)
        {
            double x = double(dst.ptr<uchar>(j)[i]);
            // 像素點為強邊緣點=255
            if (x > high)
            {
                dst.ptr<uchar>(j)[i] = 0;
            }
            // 像素點=0，被抑制掉
            else if (x < low)
            {
                dst.ptr<uchar>(j)[i] = 255;
            }
        }
    }

    // 弱邊緣點補充連接強邊緣點
    doubleThresholdLink(dst);
}

int main()
{
    Mat img = imread("test.jpg", IMREAD_GRAYSCALE); //載入灰階圖片

    // 讀取圖片失敗，則停止
    if (img.empty())
    {
        printf("讀取失敗");
        return 0;
    }
    stopWatch timer[5];
    // 高斯濾波
    timer[0].start();
    timer[1].start();
    Mat gauss_img;
    gaussianFilter(img, gauss_img); //高斯濾波器
    timer[1].stop();

    // 用一階偏導有限差分計算梯度幅值和方向
    timer[2].start();
    Mat gradXY, theta;
    getGrandient(gauss_img, gradXY, theta);
    timer[2].stop();

    // 局部非極大值抑制
    timer[3].start();
    Mat local_img;
    nonLocalMaxValue(gradXY, theta, local_img);
    timer[3].stop();

    // 用雙閥值算法檢測和連接邊緣
    timer[4].start();
    Mat dst;
    doubleThreshold(40, 80, local_img, dst);
    timer[4].stop();
    timer[0].stop();

    // 圖像顯示
    Mat outImg;
    mergeImg(outImg, img, dst); //圖像拼接
    // namedWindow("img");
    // imshow("img", outImg); // 圖像顯示
    imwrite("testcanny.jpg", dst);
    cout
        << timer[0].elapsedTime() << ", "
        << timer[1].elapsedTime() << ", "
        << timer[2].elapsedTime() << ", "
        << timer[3].elapsedTime() << ", "
        << timer[4].elapsedTime() << endl;

    return 0;
}