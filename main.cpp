#define _CRT_SECURE_NO_DEPRECATE
#define FEATURE_VECTOR_SIZE 2720
#define TRAINING_NUM_FAKES 25
#define TRAINING_NUM_REAL 28
#define CLASSES 2
#define TRAINING_SAMPLES 53
#define TEST_SAMPLES 27 
#define TEST_NUM_FAKES 10
#define TEST_NUM_REAL 17

#include <iostream>
#include <fstream>
#include <Windows.h>
#include <vector>
#include "photos.cpp"
#include "features.cpp"
#include "kNN.cpp"
//#include "subject.cpp"

#include <omp.h>

using namespace std;

#include <opencv\cv.h>
#include <opencv2\core\core.hpp>
#include <opencv\highgui.h>
#include <opencv\ml.h>

using namespace cv;

int xVals[68];
int yVals[68];

int num_of_fake_images;
int num_of_real_images;

vector<Mat> landmarks;
vector<Point2f> points;
vector<Mat> Real_Dataset;
vector<Mat> Fake_Dataset;
vector<SubjectX> subjects;
vector<SubjectX> testData;

vector<SubjectX> load_Up_Subject_Data(String, String, String);
Mat landmarkExtraction(vector<Mat>);
void learn(void);

int main(int argc, char** argv)
{
	String directory1 = "../../Image Database/smiles/smiles [original]/genuine [train]/";
	String directory2 = "../../Image Database/smiles/smiles [original]/fake [train]/";

	clock_t start;
	double duration;

	start = clock();

	subjects = load_Up_Subject_Data(directory1, directory2, "Smiles");

	directory1 = "../../Image Database/smiles/smiles [original]/genuine [test]/";
	directory2 = "../../Image Database/smiles/smiles [original]/fake [test]/";
	testData = load_Up_Subject_Data(directory1, directory2, "Smiles");

	

	cout << "^^^^^The number of subjects is " << subjects.size() << endl;
	
	
	cout << "*******TRAINING INITIALIZING*******" << endl;
    learn();
	cout << "*******TRAINING SUCCESSFUL!*******" << endl;
   
   
	cout << "\n++++++++++++EXECUTED SUCCESSFULLY!!!++++++++++\a\a\a\a" << endl;
    
	duration = (clock() - start)/(double) CLOCKS_PER_SEC;

	cout << "\n~~~~~Time taken " << duration/60 << " minutes~~~~~~" << endl;
	system("pause");
	waitKey(0);

	return 0;
}

vector<SubjectX> load_Up_Subject_Data(String dir1,String dir2, String windName)
{
	Images imgs(dir1, dir2);

	num_of_fake_images = imgs.getFakeNum();
    num_of_real_images = imgs.getRealNum();
	cout << "Fake smiles = " << num_of_fake_images << " : Real smiles = " << num_of_real_images << endl;

	vector<SubjectX> subs = imgs.getSubjects();
	vector<SubjectX> dataset;

	Features fts;

	cout << "*************************************************" << endl;
	cout << "*************************************************" << endl;

	for(size_t subject = 0; subject != subs.size(); subject++)
	   {
		  cout << "\n~~~~~~~~~SUBJECT #" << subject << "~~~~~~~~~" << endl;

	      SubjectX subX = subs.at(subject); 
          Mat subImage = subX.getImage();

	      //cout << endl;
	      ///cout << "Filtering image for subject #" << subject << endl;
	      vector<Mat> subFImgs = fts.filter(subImage);
	      //cout << "Filtering image for subject #" << subject << " done!" << endl;

          subX.setFImages(subFImgs);
	      //cout << "Loading landmarks data" << endl;
          points = subX.getLandmarks();
	      //cout << "Landmarks data loaded successfully!" << endl;
	   
	      //cout << "Extracting landmarks features " << endl;
	      Mat imgdata = landmarkExtraction(subFImgs);
	      //cout << "Landmarks feature extraction successful" << endl;
          subX.setImageData(imgdata);
	  
	      dataset.push_back(subX);
	   }
   
	cout << endl;

	return dataset;
}

Mat landmarkExtraction(vector<Mat> subFImgs)
{
	 Mat image;
	 Mat imgdata(68, 40, CV_8U);

	 Scalar in;

	 for(size_t fimage = 0; fimage != subFImgs.size(); fimage++)
	   {
		   image = subFImgs.at(fimage);
		  
		   for(size_t lmarks = 0; lmarks != points.size(); lmarks++)
		   {
			   in = image.at<uchar>(points.at(lmarks));

			   imgdata.at<uchar>(lmarks, fimage) = (uchar) in.val[0];
		   }
	   }

	 return imgdata;
}

void learn()
{
	Mat train_dataset; 
	Mat train_labelset;            // for kNN
	//Mat train_labelset(TRAINING_SAMPLES, CLASSES, CV_32F);
	Mat train_set_class(TRAINING_SAMPLES, CLASSES, CV_32F);
	Mat test_dataset(TEST_SAMPLES, FEATURE_VECTOR_SIZE, CV_32F);
	Mat test_set_class(TEST_SAMPLES, CLASSES, CV_32F); 
	

	//namedWindow("Subject image data", CV_WINDOW_AUTOSIZE);
	//namedWindow("Subject reshaped image data", CV_WINDOW_AUTOSIZE);

	cout << "Subject list size " << subjects.size() << endl;


	cout << "*******CREATING TRAINING DATASET*******" << endl;
	

	for(size_t subject = 0; subject != subjects.size(); subject++)
	{
		cout << "Subject #" << subject << endl;

		SubjectX subX = subjects.at(subject);
		Mat imdata = subX.getImageData();

		/*imshow("Subject image data", imdata);
		waitKey(999);*/
		
		Mat float_data;
		imdata.convertTo(float_data, CV_32FC1);

		cout << "\tReshaping..." << endl;
		Mat imfloat_reshaped = float_data.reshape(1, 1);

		train_dataset.push_back(imfloat_reshaped);
		
		int label = subX.getSubjectLabel();
		
		cout << subject << ": " << TRAINING_SAMPLES << endl;
		if(label == 1)
		{
			//train_labelset.at<float>(subject, 1) = 1.0;
			train_labelset.push_back(label);
			train_set_class.at<float>(subject, 1) = 1.0;
		}
		else
			if(label == -1)
			{
				//train_labelset.at<float>(subject, 0) = 1.0;
				train_labelset.push_back(label);
				train_set_class.at<float>(subject, 0) = 1.0;
			}
/*
		if(subject < (size_t) NUM_REAL)
		{
			train_labelset.push_back(1);
			train_set_class.at<float>(subject, 1) = 1.0;
		}
		else
		{
			train_labelset.push_back(0);
			train_set_class.at<float>(subject, 0) = 1.0;
		}*/
	}

	// for test kNN test data
	Mat kNN_testset;

	for(size_t test_subject = 0; test_subject != testData.size(); test_subject++)
	{
		SubjectX testSub = testData.at(test_subject);
		Mat testSub_imdata = testSub.getImageData();

		Mat float_data;
        testSub_imdata.convertTo(float_data, CV_32FC1);
	    //cout << "\tReshaping..." << endl;
	    Mat imfloat_reshaped = float_data.reshape(1, 1);

		kNN_testset.push_back(imfloat_reshaped);

		int label = testSub.getSubjectLabel();
		
		if(label == 1)
		{
			test_set_class.at<float>(test_subject, 1) = 1.0f;
		}
		else
			if(label == -1)
			{
				test_set_class.at<float>(test_subject, 0) = 1.0f;
			}

	}
	
	

	cout << "Size of train_data is " << train_dataset.size() << endl;
	cout << "Size of train_label is " << train_labelset.size() << endl << endl;
	cout << "Size of test_data is " << testData.size() << endl << endl;
	cout << "Size of test data sample input is " << kNN_testset.size() << endl;

	cout << "*******TRAINING DATASET CREATED SUCCESFULLY!*******" << endl;

	

/*
	cout << "*******CREATING TESTING DATASET*******" << endl;

	cout << "*******TESTING DATASET CREATED SUCCESSFULLY!*******" << endl;*/
	
	cout << "*******CREATING CLASSIFICATION SYSTEMS!*******\a\a\a" << endl;
    CvANN_MLP  mlp;

	Mat layerSize(3, 1, CV_32SC1);   // Create the number of layers in the neural network
	layerSize.row(0) = Scalar(FEATURE_VECTOR_SIZE);  // Number of neurons in the input layer
	layerSize.row(1) = Scalar(1361);    // Number of neurons in the 1 hidden layer
	layerSize.row(2) = Scalar(2);    // Number of neurons in the output layer

	CvANN_MLP_TrainParams params;
    CvTermCriteria criteria;

	criteria.max_iter = 130;
    criteria.epsilon = 0.0000001;
    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
    params.train_method = CvANN_MLP_TrainParams :: BACKPROP;
    params.bp_dw_scale = 0.05;
    params.bp_moment_scale = 0.05;
    params.term_crit = criteria ;
	

	mlp.create(layerSize,CvANN_MLP::SIGMOID_SYM, 0.01, 1);

	//Mat train_set_class(train_data.rows, 2, CV_32F);

	//KNearest knn;

	cout << "*******CLASSIFICATION SYSTEMS CREATED SUCCESSFULLY!\a\a\a" << endl << endl;

/*
	cout << "*******TRAINING k-NEAREST NEIGHBOURS LEARNING ALGORITHM!*******" << endl << endl;

	bool check = knn.train(train_dataset, train_labelset, Mat(), false, 53, false);

	if(check)
	{
		cout << "\tTRAINING k-NEAREST NEIGHBOURS LEARNING ALGORITHM SUCCESSFULL!\a\a\a" << endl << endl;
	}
	else
	{
		cout << "\tTRAINING k-NEAREST NEIGHBOURS LEARNING ALGORITHM FAILED!\a\a\a" << endl << endl;
	}

	cout << "*******TESTING THE k-NEAREST NEIGHBOUR ALGORITHM!*******" << endl;

	/*Mat sample;
	SubjectX sub = testData.at(2);
	Mat imdata = sub.getImageData();
	
	String tag = sub.getTag();
	namedWindow(tag, CV_WINDOW_AUTOSIZE);
	*/
	//for(int K = 3; K < 28; K += 2){
	/*int K = 25;/*
	Mat nearests(1, K, CV_32FC1);

	Mat float_data;
    imdata.convertTo(float_data, CV_32FC1);
	cout << "\tReshaping..." << endl;
	Mat imfloat_reshaped = float_data.reshape(1, 1);*

	Mat results(TEST_SAMPLES, 1, CV_32F);
	//Mat neighbours(1, K, CV_32F);
	Mat neighboursResponse(TEST_SAMPLES, K, CV_32F);
	Mat dist(TEST_SAMPLES, K, CV_32F);


	//sample.push_back(imfloat_reshaped);

	float response = knn.find_nearest(kNN_testset, K, results, neighboursResponse, dist);

	//cout << "The label of the test subject -> " << sub.getSubjectLabel() << endl;
	//cout << "The value of response is -> " << response << endl;
	cout << "The value of K is -> " << K << endl;
	//cout << "The value of results (the class of the sample) -> " << results << endl;
	//cout << "The neighbourResponse -> " << neighboursResponse << endl;
	//cout << "The dist -> " << dist << endl;

	namedWindow(subjects.at(0).getTag(), CV_WINDOW_AUTOSIZE);
	namedWindow(subjects.at(subjects.size() - 1).getTag(), CV_WINDOW_AUTOSIZE);

	int fsmilenumP = 0;
	int rsmilenumP = 0;

	for(int res = 0; res < results.rows; res++)
	{
		if(res < TEST_NUM_REAL)
		{
			if(results.at<float>(res, 0) == -1)
	    {
		   cout << "*******THE SMILE IS NON-SPONTANEOUS*******!\a\a\a\a" << endl;
	    }
	    else
		   if(results.at<float>(res, 0) == 1)
		   {
			 cout << "*********THE SMILE IS SPONTANEOUS**********!" << endl;
			  rsmilenumP++;
		   }
		}
		else
		{
           if(results.at<float>(res, 0) == -1)
	       {
		      cout << "*******THE SMILE IS NON-SPONTANEOUS*******!\a\a\a\a" << endl;
			  fsmilenumP++;
	       }
	       else
		      if(results.at<float>(res, 0) == 1)
		      {
			     cout << "*********THE SMILE IS SPONTANEOUS**********!" << endl;
		      }
		}
		

	   SubjectX test_sub = testData.at(res);
	   Mat test_subimage = test_sub.getImage();
	   
	   String tag = test_sub.getTag();
	   int label = test_sub.getSubjectLabel();

	   cout << "Test image actual class -> " << label << endl;
	   cout << "Test image predicted class -> " << results.at<float>(res, 0) << endl;


	   imshow(tag, test_subimage);
	   waitKey(0);
	}

	
	cout << "*********************************************************************************" << endl;
	cout << "Fake smile detection accuracy ----> " << ((double) fsmilenumP/(double) TEST_NUM_FAKES) * 100 << endl;
	cout << "Real smile detection accuracy ----> " << ((double) rsmilenumP/(double) TEST_NUM_REAL) * 100 << endl;
    cout << "*********************************************************************************" << endl;
	//}
	/*imshow(tag, sub.getImage());
	waitKey(500);*/
	
	cout << "*******TRAINING MULTI-LAYERED PERCEPTRON LEARNING ALGORITHM!*******" << endl << endl;
	/*#define TRAINING_SAMPLES 3050       //Number of samples in training dataset
    #define ATTRIBUTES 256  //Number of pixels per sample.16X16
    #define TEST_SAMPLES 1170       //Number of samples in test dataset
    #define CLASSES 10                  //Number of distinct labels.
	//matrix to hold the training sample
    cv::Mat training_set(TRAINING_SAMPLES,ATTRIBUTES,CV_32F);
    //matrix to hold the labels of each taining sample
    cv::Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);
    //matric to hold the test samples
    cv::Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_32F);
    //matrix to hold the test labels.
    cv::Mat test_set_classifications(TEST_SAMPLES,CLASSES,CV_32F);**/

	

    //int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
	int num_of_iterations = mlp.train(train_dataset,train_set_class,Mat(),Mat(),params);
	cout << "Number of iterations taken in training the neural network -> " << num_of_iterations << endl << endl;
	cout << "*******TRAINING MULTI-LAYERED PERCEPTRON LEARNING ALGORITHM SUCCESSFUL!*******" << endl << endl;
	
	cout << "*******TESTING DATASET SAMPLE!*******" << endl << endl;
	/*
	Mat classResult(1, CLASSES, CV_32F);
	Mat test_sample;

	// count of correct classification
	int correct_class = 0;
	// count of incorrect classification
	int wrong_class = 0;

	// classification matrix gives the count of classes to which the samples were classified
	int classification_matrix[CLASSES][CLASSES] = {{}};

	// for each sample in the  test set.
	for(int tsample = 0; tsample < TEST_SAMPLES; tsample++)
	{
		// extract the sample
		test_sample = kNN_testset.row(tsample);

		// try to predict its class
		mlp.predict(test_sample, classResult);

		/*The classification result matrix holds weightage  of each class.
            we take the class with the highest weightage as the resultant class 
		
		// find the class with with maximum weightage
		int maxIndex = 0;
		float value = 0.0f;
		float maxValue = classResult.at<float>(0, 0);

		for(int index = 1; index < CLASSES; index++)
		{
			value = classResult.at<float>(0, index);

			if(value > maxValue)
			{
				maxValue = value;
				maxIndex = index;
			}
		}

		printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);
 
        //Now compare the predicted class to the actural class. if the prediction is correct then\
        //test_set_classifications[tsample][ maxIndex] should be 1.
        //if the classification is wrong, note that.
        if(test_set_class.at<float>(tsample, maxIndex)!=1.0f)
        {
           // if they differ more than floating point error => wrong class
 
           wrong_class++;
 
           //find the actual label 'class_index'
           for(int class_index=0;class_index<CLASSES;class_index++)
           {
              if(test_set_class.at<float>(tsample, class_index)==1.0f)
              {
                 classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
                 break;
              }
           }
        } 
		else
		{
           // otherwise correct
           correct_class++;
           classification_matrix[maxIndex][maxIndex]++;
        }
     }
 
     printf( "\nResults on the testing dataset\n""\tCorrect classification: %d (%g%%)\n""\tWrong classifications: %d (%g%%)\n", 
             correct_class, (double) correct_class*100/TEST_SAMPLES,
             wrong_class, (double) wrong_class*100/TEST_SAMPLES);
     cout << "   ";
     for (int i = 0; i < CLASSES; i++)
     {
        cout<< i<<"\t";
     }
     cout<<"\n";
     for(int row=0;row<CLASSES;row++)
     {
		 cout<< row << "  ";
         for(int col=0;col<CLASSES;col++)
         {
            cout<<classification_matrix[row][col]<<"\t";
         }
         
		 cout<<"\n";
     }*/
	
	Mat predicted(kNN_testset.rows, CLASSES, CV_32F);

	for(int test_sub = 0; test_sub < kNN_testset.rows; test_sub++)
	{
		Mat response(1, CLASSES, CV_32F);
		Mat sample = kNN_testset.row(test_sub);
		mlp.predict(sample, response);

		for(int category = 0; category < CLASSES; category++)
		{
			predicted.at<float>(test_sub, category) = response.at<float>(0, category);
		}
	}

	namedWindow(testData.at(0).getTag(), CV_WINDOW_AUTOSIZE);
	namedWindow(testData.at(0).getTag(), CV_WINDOW_AUTOSIZE);

	
	for(int test_sub = 0; test_sub < testData.size(); test_sub++)
	{
		SubjectX sub = testData.at(test_sub);
		Mat test_image = sub.getImage();

		imshow(sub.getTag(), test_image);
		waitKey(0);

		float maxWeight = predicted.at<float>(test_sub, 0);
		int maxWIndex = 0;
		float tempWeight = 0.0f;

		for(int index = 1; index < CLASSES; index++)
		{
			tempWeight = predicted.at<float>(test_sub, index);

			if(tempWeight > maxWeight)
			{
				maxWeight = tempWeight;
				maxWIndex = index;
			}
		}


		cout << "***********************************" << endl;
		cout << "The actual label of the subject " << sub.getSubjectLabel() << endl;

		if(maxWIndex == 0)
		{
			cout << "The predicted label of the subject " << -1 << endl;
		}
		else
			if(maxWIndex == 1)
			{
			   cout << "The predicted label of the subject " << 1 << endl;
			}
		
	   imshow(sub.getTag(), sub.getImage());
	   waitKey(999);
	}


	
	cout << "*******TESTING DATASET SAMPLE SUCCESSFUL!\a\a\a*******" << endl << endl;

	cout << "*******THE CONFUSION MATRIX PART*******\a\a\a" << endl << endl;
	
	cout << "*********************************************************" << endl << endl;
}
/*
void displayMulti()
{
   //Image Reading
   IplImage* img1 = cvLoadImage( "../../Image Database/smiles/genuine smiles/S032_006_1.png" );
   IplImage* img2 = cvLoadImage( "../../Image Database/smiles/genuine smiles/S035_006_1.png" );
   IplImage* img3 = cvLoadImage( "../../Image Database/smiles/genuine smiles/S034_005_1.png" );
   IplImage* img4 = cvLoadImage( "../../Image Database/smiles/genuine smiles/S065_004_1.png" );
   int dstWidth=img1->width+img1->width;
   int dstHeight=img1->height+img1->height;

   IplImage* dst=cvCreateImage(cvSize(dstWidth,dstHeight),IPL_DEPTH_8U,3); 

   // Copy first image to dst
   cvSetImageROI(dst, cvRect(0, 0,img1->width,img1->height) );
   cvCopy(img1,dst,NULL);
   cvResetImageROI(dst);

   // Copy second image to dst
   cvSetImageROI(dst, cvRect(img2->width, 0,img2->width,img2->height) );
   cvCopy(img2,dst,NULL);
   cvResetImageROI(dst);

   // Copy third image to dst
   cvSetImageROI(dst, cvRect(0, img3->height,img3->width,img3->height) );
   cvCopy(img3,dst,NULL);
   cvResetImageROI(dst);

   // Copy fourth image to dst
   cvSetImageROI(dst, cvRect(img4->width, img4->height,img4->width,img4->height) );
   cvCopy(img4,dst,NULL);
   cvResetImageROI(dst);

   //show all in a single window
   cvNamedWindow( "Example1", CV_WINDOW_AUTOSIZE );
   cvShowImage( "Example1", dst );
   cvWaitKey(0); 
}*/


 