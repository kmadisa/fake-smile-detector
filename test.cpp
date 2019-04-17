#include <iostream>
#include <vector>
#include <fstream>
#include "photos.cpp"

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <opencv/cv.h>

using namespace cv;

bool is_file_exist(String);
/*
int main()
{
	/*Mat image = imread("../../Image Database/smiles/counterfeit smiles/S022_003_1.png", CV_LOAD_IMAGE_GRAYSCALE);

	bool y;

	struct stat st;
	String tmp = "../../Image Database/smiles/counterfeit smiles/S022_003_1.png";
	
	y = is_file_exist(tmp);

	cout << "The file exist: " << y << endl;

	/*cout << "Number of channels " << image.channels() << endl;
	namedWindow("Image", CV_WINDOW_AUTOSIZE);
	imshow("Image", image);

	cout << "Image columns " << image.cols << endl;
	cout << "Image rows " <<  image.rows << endl;
	cout << "Image type " << image.type() << endl;

	Mat tstimg(image.rows, image.cols, CV_8U);

	cout << tstimg.size() << endl;

	Scalar im;

	for(int row = 0; row < tstimg.rows; row++)
	{
		for(int col = 0; col < tstimg.cols; col++)
		{
			im = image.at<uchar>(row, col);
			tstimg.at<uchar>(row, col) = im.val[0];
		}

	}

	//cout << tstimg << endl;

	namedWindow("Reconstructed image", CV_WINDOW_AUTOSIZE);
	imshow("Reconstructed image", tstimg);**
	String temp = "../../Image Database/smiles/counterfeit smiles/";
	Images imgs(temp, "Fake Smiles");

	Mat im = imgs.getImage(23);

	namedWindow("Tsting", CV_WINDOW_AUTOSIZE);
	imshow("Tsting", im);
	system("pause");
	waitKey(0);

	return 0;
}*/

bool is_file_exist(String fileName)
{
	ifstream infile(fileName);

	return infile.good();
}