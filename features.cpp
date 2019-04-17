#include <iostream>
#include <vector>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv\ml.h>
#include <opencv\cv.h>

using namespace cv;

class Features
{
   public: string favoriteProgram;
   private: vector<Mat> filters;
   private: vector<Mat> fImgs;
   private:  vector<Mat> fImg;

   public: Features()
		   {
			   cout << "Creating gabor filters" << endl;
			   createFilters();
			   cout << "Gabor filters created successfully!!!" << endl;
		   }


   public: void displayGaborEnergyFilters()
			{
				cout << " Displaying filters" << endl;
				int numOrientations = 8;
                int n = 0;
				double degrees = 0.0;
				
				namedWindow("Gabor energy filters", CV_WINDOW_NORMAL);

				for(int filNum = 0; filNum < numOrientations; filNum++, degrees += 22.5)
				{
					for(int scales = 0; scales < 5; scales++)
					{
						Mat kernel = filters.at(n);
						n++;

						cout << "Orientation: " << degrees << "  frequency scale : #" << scales << endl;
						imshow("Gabor energy filters", kernel);
					    waitKey(800);
					}

				}
			}

   public: void testOnImages()
			{
				Mat dest, src_f;
	            Mat image = imread("../../Image Database/smiles/counterfeit smiles/S022_003_1.png", CV_LOAD_IMAGE_GRAYSCALE);
	            imshow("Loaded image", image);
				equalizeHist(image, image);

	            for(int x = 0; x < filters.size(); x++)
	            {
		            image.convertTo(src_f,CV_32F);
					Mat kernel = filters.at(x);
	                filter2D(src_f, dest, CV_32F, kernel);

                    Mat viz;
                    dest.convertTo(viz,CV_8U,1.0/255.0);
                    imshow("Gabor filtered image",viz);
		            waitKey(800);
		            cout << "Gabor filtered image " << x << endl;
	            }


	           cout << "Done testing ..." << endl;
			}

   private: void createFilters()
		   {
			   vector<Mat> gFilters;
			   
			   double ratios[] = {0.3776, 0.7054, 1.3854, 2.7580, 5.5096};
               double sigg[5];
			   double sigma = 90, theta = 0, lambd = 29.0, gamma = 0.5, psi = 0.0;

			   int siglen = sizeof(sigg)/sizeof(double);

			   for(int x = 0; x < siglen; x++)
			   {
				   sigg[x] = ratios[x] * lambd;
			   }

			   Mat kernel;
			   int kernel_size = 30;

			   for(double x = 0; x < 8; x++, theta += 22.5)
			   {
				   for(int y = 0; y < 5; y++)
				   {
					   kernel = getGaborKernel(Size(kernel_size,kernel_size), sigg[y], theta, lambd, gamma, psi);

	                   gFilters.push_back(kernel);
				   }
				 
			   }

			   filters = gFilters;
		   }


   public: vector<Mat> getFilters()
		   {
			   return filters;
		   }

   public: vector<Mat> filter(Mat image)
		   {
			   Mat dest, src_f, kernel;
			   //namedWindow("Filtered", CV_WINDOW_AUTOSIZE);
			   vector<Mat> filteredImgs;

			   for(int filter = 0; filter < filters.size(); filter++)
			   {
				   kernel = filters.at(filter);
                  
				   image.convertTo(src_f,CV_32F);
	               filter2D(src_f, dest, CV_32F, kernel);
					   
				   Mat viz;
                   dest.convertTo(viz,CV_8U,1.0/255.0);

				   
				   filteredImgs.push_back(viz);
			   }

			   return filteredImgs;
		   }

   public: vector<Mat> getFilteredImg()
		   {
			   return fImg;
		   }

};

/*int main()
{
	waitKey(0);
}*/