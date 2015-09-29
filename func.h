#ifndef __func__
#define __func__

#include "const.h"

using namespace cv;
/*------------------------------------*/
/*    Calculate background            */
/*------------------------------------*/
IplImage* bg(int index, int nb_cam, int start, int end);


/*-----------------------------------------------------------------------*/
/*      Component Connection Algorithem  provided by Mohamed Dahmane     */
/*-----------------------------------------------------------------------*/
int etiquetage(IplImage* imageBin, CvMat* &matEtiq);
void convertEtiq(CvMat* matEtiq, IplImage* &imgEtiq, int nbEtiq);


/*-----------------------------------------------*/
/*    Select the largest segmentation            */
/*-----------------------------------------------*/
void select(IplImage* imgEtiq, float &xmax, float &xmin, float &ymax, float &ymin );


/*----------------------------------------------*/
/*   Calculate  the ratio: width/height         */
/*----------------------------------------------*/
float ratio1(float xmax, float xmin, float ymax, float ymin);


/*------------------------------------------------*/
/*   Calculate the change velocity of ratio       */
/*------------------------------------------------*/
void change(float* tmpBufferRatio, int nb );


/*-------------------------------------------------------------------------------------*/
/*        Compute PDF of Gaussian  provided by http://code.google.com/p/opencvx/       */
/*-------------------------------------------------------------------------------------*/
void cvMatGaussPdf( CvMat* samples, CvMat* mean, CvMat* cov, CvMat* probs, bool normalize, bool logprob);
float cvGaussPdf( CvMat* sample, CvMat* mean, CvMat* cov, bool normalize, bool logprob);


/*-----------------------------------------------------------------------------------------------*/
/*        the statistics of data:   nb of frames of each scenes captured by one camera           */ 
/*                                  total nb of frames captured by one camera                    */
/*                                  the index of image begining and ending in each scenes        */
/*-----------------------------------------------------------------------------------------------*/
void statis(int start_scene, int end_scene, int* start_frames, int* end_frames, int* total_frames, int nb_cam, int &sum_nb);


/*-----------------------------------------------------------*/
/*        Load all the data captured by each camera          */
/*-----------------------------------------------------------*/
void loadData (int start_scene, int end_scene, int* start_bg, int* end_bg, int* start_frames, int* end_frames, 
                               int* total_frames, float* label, int nb_cam, int sum_nb, point* frame );


/*----------------------------------------------------------*/
/*        Estimate parametres of mixtured Gaussian          */
/*----------------------------------------------------------*/
void paramEstim (point* frames, point &mean0, point &mean1, 
                    Mat var0, Mat var1, float &p0, float &p1, float* label, int sum_nb );


/*----------------------------------------------------------*/
/*                Training Cameras                          */
/*----------------------------------------------------------*/
void TrainingCam1(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1);
void TrainingCam2(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1);
void TrainingCam3(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1);
void TrainingCam4(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1);


#endif