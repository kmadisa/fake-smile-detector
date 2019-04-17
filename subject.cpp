#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <opencv/cv.h>


using namespace cv;


class SubjectX
{
   public: String imageID;
			String lmarkFileName;
			String directory;
			Mat image;
			Mat image_AMM;
			Mat data;
			vector<Mat> fImages;
			vector<Point2f> landmarks;
			int label;

   public: SubjectX(){}
   public: SubjectX(String dir,String imgID, String landmarksID, int num)
		   {
			   directory = dir;
			   imageID = imgID + ".png";
			   lmarkFileName = landmarksID + ".txt";
			   label = num;

			   ///cout << directory + imageID << endl;
			   //cout << directory + lmarkFileName << endl;
               loadImage();
			   loadAAMs();
		   }

   private: void loadImage()
		   {
			   //namedWindow(directory, CV_WINDOW_AUTOSIZE);
			   Mat tempimage = imread(directory + imageID, CV_LOAD_IMAGE_GRAYSCALE);
			   normalize(tempimage, image, 0, 255, CV_MINMAX);
			  // imshow(directory, image);
			  // waitKey(400);
		   }
  
   private: void loadAAMs()
		   {
			   char xVal[15], yVal[15];
	           int x, y;

	           ifstream in(directory + lmarkFileName);

	           if(!in)
	           {
                   cout << "Cannot open file.\n";
	               exit(1);
	           }

	           int index = 0;

	           do
	           {
	              in >> xVal;
	              in >> yVal;

	              x = (int) atof(xVal);
	              y = (int) atof(yVal);

	              Point cord(x, y);

	              landmarks.push_back(cord);
 
	              index++;

	           }while((in != NULL) & (index < 68));
		   }

   public: Mat getImage()
		   {
              return image;
		   }

   public: vector<Mat> getFImages()
		   {
			   return fImages;
		   }

   public: void setFImages(vector<Mat> fimages)
		   {
			   fImages = fimages;
		   }

   public: vector<Point2f> getLandmarks()
		   {
			   return landmarks;
		   }

   public: void setImageData(Mat lmkImage)
		   {
			  data = lmkImage;
		   }

   public: Mat getImageData()
		   {
              return data;
		   }

   public: int getSubjectLabel()
		   {
			   return label;
		   }
 
   public: String getTag()
		   {
			   return directory;
		   }

};

