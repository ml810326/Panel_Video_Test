#include <iostream>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cv.h>
#include <string>
#include <fstream>
#include <string>
#include <omp.h>
#include <math.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

//Defining PI value
#ifndef PI
 #ifdef M_PI
  #define PI M_PI
 #else
  #define PI 3.14159265358979
 #endif
#endif

using namespace cv;
using namespace std;

// calculate entropy of an image
double Entropy(Mat img){
	double temp[256];
	for(int i=0;i<256;i++){
		temp[i] = 0.0;
	}
	for(int m=0;m<img.rows;m++){
		const uchar* t = img.ptr<uchar>(m);
		for(int n=0;n<img.cols;n++){
			int i = t[n];
			temp[i] = temp[i]+1;
		}
	}
	for(int i=0;i<256;i++){
		temp[i] = temp[i]/(img.rows*img.cols);
	}
	double result = 0;
	for(int i =0;i<256;i++){
		if(temp[i]==0.0)
			result = result;
		else
			result = result-temp[i]*(log(temp[i])/log(2.0));
	}
	return result;
}

void BGRtoY(Mat input, Mat Y){
	Y = Mat::zeros(input.rows,input.cols,CV_8UC1);
	double R=0.0, B=0.0, G=0.0;
	for(int j = 0;j < input.rows; j++){
		for(int i = 0;i < input.cols; i++){
			B=(double)(input.at<Vec3b>(j,i)[0]);	
			G=(double)(input.at<Vec3b>(j,i)[1]);	
			R=(double)(input.at<Vec3b>(j,i)[2]);

			Y.ptr<uchar>(j)[i] = (uchar)(0.299*R+0.587*G+0.114*B);
		}
	}
}

void fastmean(Mat input,double men[3],float ga){
	double R=0.0, B=0.0, G=0.0, temp=0.0;
	double pop = input.rows*input.cols;
	float va = 1/ga;
	int vara = va;

	for(int i = 0;i < pop; i=i+va){
		temp++;
		B+=(double)(input.data[i*3]);	
		G+=(double)(input.data[i*3+1]);	
		R+=(double)(input.data[i*3+2]);
	}
	men[0] = R/temp;
	men[1] = G/temp;
	men[2] = B/temp;
}


void pwr_enhancerV2(Mat input, Mat output, double mu[3], double para[4], double t){//[0] mean [1] OR_th [2] Air [3]Sd

	double deltaR = 0.0, deltaG = 0.0, deltaB = 0.0, I=0.0, T = 0.0, localad = 0.0;
	deltaR = mu[0] * para[0];
	deltaG = mu[1] * para[0];
	deltaB = mu[2] * para[0];

	double R, G, B;
	for (int j = 0; j < input.rows; j++){
		for (int i = 0; i < input.cols; i++){
			I = (0.299*(double)input.at<Vec3b>(j,i)[2] + 0.587*(double)input.at<Vec3b>(j,i)[1] + 0.114*(double)input.at<Vec3b>(j,i)[0]);
			I = floor(I+0.5);
			T = para[0];
			if (I<=T){
				localad = (pow((I/255),(255/T))+((T/255)/2));
			}
			else{
				localad = (pow((T/255),(255/T))+((T/255)/2)) + (pow((I/255),(255/T)) - pow((T/255),(255/T)))*(2*exp(((para[3]/100)-0.5)));
			}
			B = t*((double)input.at<Vec3b>(j,i)[0] - localad*deltaB); 
			if (B < 0.0){B = 0.0;}
			else if (B > 255.0){B = 255.0;} else{}
			output.at<Vec3b>(j,i)[0] = (uchar)(cvRound(B));
			G = t*((double)input.at<Vec3b>(j,i)[1] - localad*deltaG); 
			if (G < 0.0){G = 0.0;}
			else if (G > 255.0){G = 255.0;} else{}
			output.at<Vec3b>(j,i)[1] = (uchar)(cvRound(G));
			R = t*((double)input.at<Vec3b>(j,i)[2] - localad*deltaR); 
			if (R < 0.0){R = 0.0;}
			else if (R > 255.0){R = 255.0;} else{}
			output.at<Vec3b>(j,i)[2] = (uchar)(cvRound(R));		
		}}
}

void pwr_enhancerfV2(Mat input, Mat output, double mu[3], double para[4], double t){//[0] mean [1] OR_th [2] Air [3]Sd
	double IR = 0.0, IG = 0.0, IB = 0.0;
	double TR = 0.0, TG = 0.0, TB = 0.0;

	#pragma omp parallel
	{
	#pragma omp sections nowait
	{
		#pragma omp section
		{
			double deltaR = 0.0, localadR = 0.0;
			deltaR = mu[0] *  para[0];
			double R, G, B;
			for (int j = 0; j < input.rows; j++){
				for (int i = 0; i < input.cols; i++){
					IR = (0.299*(double)input.at<Vec3b>(j,i)[2] + 0.587*(double)input.at<Vec3b>(j,i)[1] + 0.114*(double)input.at<Vec3b>(j,i)[0]);
					IR = floor(IR+0.5);
					TR = para[0];
					if (IR<=TR){
						localadR = (pow((IR/255),(255/TR))+((TR/255)/2));
					}
					else{
						localadR = (pow((TR/255),(255/TR))+((TR/255)/2)) + (pow((IR/255),(255/TR)) - pow((TR/255),(255/TR)))*(2*exp(((para[3]/100)-0.5)));
					}
					R = t*((double)input.at<Vec3b>(j,i)[2] - localadR*deltaR); 
					if (R < 0.0){R = 0.0;}
					else if (R > 255.0){R = 255.0;} 
					else{}
					output.at<Vec3b>(j,i)[2] = (uchar)(cvRound(R));		
				}
			}
		}
		#pragma omp section
		{
			double deltaG = 0.0, localadG = 0.0;
			deltaG = mu[1] *  para[0];
			double R, G, B;
			for (int j = 0; j < input.rows; j++){
				for (int i = 0; i < input.cols; i++){
					IG = (0.299*(double)input.at<Vec3b>(j,i)[2] + 0.587*(double)input.at<Vec3b>(j,i)[1] + 0.114*(double)input.at<Vec3b>(j,i)[0]);
					IG = floor(IG+0.5);
					TG = para[0];
					if (IG<=TG){
						localadG = (pow((IG/255),(255/TG))+((TG/255)/2));
					}
					else{
						localadG = (pow((TG/255),(255/TG))+((TG/255)/2)) + (pow((IG/255),(255/TG)) - pow((TG/255),(255/TG)))*(2*exp(((para[3]/100)-0.5)));
					}
					G = t*((double)input.at<Vec3b>(j,i)[1] - localadG*deltaG); 
					if (G < 0.0){G = 0.0;}
					else if (G > 255.0){G = 255.0;} 
					else{}
					output.at<Vec3b>(j,i)[1] = (uchar)(cvRound(G));
				}
			}
		}
		#pragma omp section
		{
			double deltaB = 0.0, localadB = 0.0;
			deltaB = mu[2] *  para[0];
			double R, G, B;
			for (int j = 0; j < input.rows; j++){
				for (int i = 0; i < input.cols; i++){
					IB = (0.299*(double)input.at<Vec3b>(j,i)[2] + 0.587*(double)input.at<Vec3b>(j,i)[1] + 0.114*(double)input.at<Vec3b>(j,i)[0]);
					IB = floor(IB+0.5);
					TB = para[0];
					if (IB<=TB){
						localadB = (pow((IB/255),(255/TB))+((TB/255)/2));
					}
					else{
						localadB = (pow((TB/255),(255/TB))+((TB/255)/2)) + (pow((IB/255),(255/TB)) - pow((TB/255),(255/TB)))*(2*exp(((para[3]/100)-0.5)));
					}
					B = t*((double)input.at<Vec3b>(j,i)[0] - localadB*deltaB); 
					if (B < 0.0){B = 0.0;}
					else if (B > 255.0){B = 255.0;}
					else{}
					output.at<Vec3b>(j,i)[0] = (uchar)(cvRound(B));
				}
			}
		}
	}}
}

void set_parameter(Mat frame, double para[4]){
	Mat gray = Mat::zeros(frame.rows,frame.cols, CV_8UC1);
	BGRtoY(frame, gray);
	
	double Air = 0.0;
	double mean = 0.0;
	double OR_th = 0.0;
	double sd = 0.0;
	double pop = (double)(frame.rows*frame.cols);
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			mean += (double)gray.ptr<uchar>(j)[i];
		}
	}
	mean /= pop;
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			OR_th += pow(((double)gray.ptr<uchar>(j)[i]-mean),2);
		}
	}
	OR_th /= pop;
	sd = sqrt(OR_th);
	OR_th = mean + (0.5*sqrt(OR_th));

	if (OR_th>255){
		OR_th = 255;
	}

	double beta = 25.0;
	
	if (OR_th>250){
		Air = OR_th;
	}
	else{
		Air = (255.0 + (OR_th-beta))/2;
	}

	para[0] = mean;
	para[1] = OR_th;
	para[2] = Air;
	para[3] = sd;
}

int main()
{
	// read video
	VideoCapture vc("C:/Users/NTUST-LiMing/Desktop/2016matlabcode/algorithm/t_video/Avatar - Eywa.mp4");
	int fps = vc.get(CV_CAP_PROP_FPS);
	int fH = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
	int fW = vc.get(CV_CAP_PROP_FRAME_WIDTH);

	//create output video
	VideoWriter vm("Power_saving_mine_Avatart.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(fW, fH));
	if (vc.isOpened() == false) {
		cout << "Fail to open video" << endl;
		system("pause");
		return 1;
	}
	if (vm.isOpened() == false) {
		cout << "Fail to write video" << endl;
		system("pause");
		return 1;
	}

	//create window
	namedWindow("show", CV_WINDOW_AUTOSIZE);
	namedWindow("output2", CV_WINDOW_AUTOSIZE);

	//RGB to Y
	Mat frame, dist, dist2;
	double mean  = 0.0, OR_th  = 0.0, Air  = 0.0, sd = 0.0, cov_rate = 0.0;
	vc >> frame;

	double E = 0.0;

	double para[4];//[0] mean [1] OR_th [2] Air [3]Sd
	set_parameter(frame, para);
	mean = para[0];
	OR_th = para[1];
	Air = para[2] + 1;
	sd = para[3];
	
	// RGB mean
	double men[3];
	// RGB parameter
	double mu[3];
	// RGB fast mean
	double menf[3];
	float ga = 0.05;
	
	char key;
	char pause;
	int count = 0;
	double tempE = Entropy(frame);
	double dE = 0.0;
	double rateE = 0.0;
	while (frame.empty() == false) 
	{
		//count frame number
		count = count + 1;
		//show oringinal image
		imshow("show", frame);

		//Acquires input
		Mat input = frame;
		Mat output2 = Mat::zeros(input.rows,input.cols,input.type());
		
		E = Entropy(frame);
		dE = E - tempE;
		if (dE<0){
			dE = -dE;
		}
		rateE = dE/(tempE+0.0001);
		if (rateE>0.05){
			tempE = E;
			set_parameter(frame, para);
			mean = para[0];
			OR_th = para[1];
			Air = para[2] + 1;
			sd = para[3];
		}

		double t = 0.0, evar = 0.0, ttemp = 0.0, etemp = 0.0, atemp = 0.0;
		evar = sd/10;
	    ttemp = 1-pow(((mean-128)/128),2);
		etemp = (((mean-128)*ttemp)/(2*pow(evar,2)));
		atemp = 5/(evar*(sqrt(2*PI)));
		t = 0.68-atemp*(exp(-(1-etemp)));

		double Rmu = 0.0, Gmu = 0.0, Bmu = 0.0;
		double ad_t = exp(-(mean/255));

		Rmu = ad_t*((Air - mean)*(1-t))/(t*mean);
		Gmu = ad_t*((Air - mean)*(1-t))/(t*mean);
		Bmu = ad_t*((Air - mean)*(1-t))/(t*mean);

		mu[0] = Rmu;
		mu[1] = Gmu;
		mu[2] = Bmu;

		pwr_enhancerfV2(input, output2, mu, para, t);

		key = waitKey(33);
		/*if(key==27)
			pause=!pause;
        while(pause)
        {
            key=waitKey(1000);
            if(key=='t')
            pause=!pause;
        }*/
		if (key == 'b')
			break;
		int maxframe = fps*60;
		if (count==maxframe)
			break;

		resize(output2, dist2, Size(fW, fH));
		imshow("output2", dist2);
		vm << dist2;

		vc >> frame;
		
	}
	return 0;
}