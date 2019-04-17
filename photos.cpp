#include <stdio.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/ml.h>
#include <opencv/cv.h>
#include "subject.cpp"

using namespace cv;

class Images
{
   private: vector<SubjectX> images;

   public: Images(string directory1, String directory2/*, string windName*/)
		    {
				cout << "Loading up subject data." << endl;
                loadImages(directory1/*, windName*/);
				loadImages(directory2);
				cout << "Subject data loaded successfully!!!" << endl;
		    }

   private: void loadImages(string directory/*, string windName*/)
			{
				char key = 'S', buff = '0', underscore = '_', subjectImageNo = '1';
	            string buffs = "00", fileExtensions = ".png";
	
	            int subjectNoMin = 10, subjectNoMax = 147, subjectNoFolder = 13;
	            String img_file_ID;
	            //string Cdirectory = "../../Image Database/smiles/genuine smiles/60x60/";
	            Mat image;
	
	            string url;
	            string fullFileName;
	            stringstream convert;    //string stream used for the conversion

	            convert << subjectNoMin;    // add the value of number to the characters in the stream 
	            string result;

	            result = convert.str();   // set result to the content of the stream

	            string dir = directory + key;
				img_file_ID = key;
	            Mat gray_image;
	            Mat resized;
	
	            //namedWindow(windName, CV_WINDOW_FREERATIO);
	            //resizeWindow(windName, 420, 396); 
	            string temp, temp1;

	            for(int imageNo = subjectNoMin; imageNo <= subjectNoMax; imageNo++)
	            {
	                convert.str(std::string());
	                convert << imageNo;
	                result = convert.str();

	                if(imageNo < 100)
	                {
	                   url = dir + buff + result;
		               temp = "S0" + result;
	                }
	                else
	                {
	                   url = dir + result;
		               temp = key + result;
	                }

	                temp += underscore;
	                url += underscore;

	                for(int folder = 1; folder <= subjectNoFolder; folder++)
	                {
	                    convert.str(std::string());
		                convert << folder;
		                result = convert.str();

		                if(folder > 9)
		                {
			               fullFileName = url + buff + result + underscore + subjectImageNo + fileExtensions;
						   img_file_ID = temp + buff + result + underscore + subjectImageNo;
			               //temp1 = temp + buff + result + underscore + subjectImageNo + fileExtensions;
		                }
		                else
		                {
			               fullFileName = url + buffs + result + underscore + subjectImageNo + fileExtensions;
						   img_file_ID = temp + buffs + result + underscore + subjectImageNo;
			               //temp1 = temp + buffs + result + underscore + subjectImageNo + fileExtensions;
		                }

		                if(is_file_exist(fullFileName))
		                {
			                SubjectX subject(directory, img_file_ID, img_file_ID);
							images.push_back(subject);
						}
					}

				}

	           //cout << "\n\n\nThe number of " << windName << " is "<< images.size() << endl << endl;

			}

   public: Mat getImage(int index)
		   {
               SubjectX sub = images.at(index);
			   Mat image = sub.getImage();

			   return image;
		   }

   public: vector<SubjectX> getSubjects()
		   {
              return images;
		   }

   private: bool is_file_exist(String fileName)
   {
	  ifstream infile(fileName);

	  return infile.good();
   }

};