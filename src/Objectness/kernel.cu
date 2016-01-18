/************************************************************************/
/* This source code is	free for both academic and industry use.         */
/* Some important information for better using the source code could be */
/* found in the project page: http://mmcheng.net/bing					*/
/************************************************************************/

#include "stdafx.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"


// Uncomment line line 14 in Objectness.cpp to remove counting times of image reading.

void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz);

int main(int argc, char* argv[])
{
	//DataSetVOC::importImageNetBenchMark();
	//DataSetVOC::cvt2OpenCVYml("C:/WkDir/DetectionProposals/VOC2007/Annotations/");

	RunObjectness("WinRecall.m", 2, 8, 2, 130);
	return 0;
}

void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz)
{
	srand(131);//srand((unsigned int)time(NULL));
	//omp_set_num_threads(16);
	DataSetVOC voc2007("/home/tolgab/workspace_db/bing/BING++/Code2/build/datasets/VOC2007/");   //"D:/WkDir/COCO/"  //"D:/WkDir/NYU/"  //"C:/WkDir/DetectionProposals/VOC2007/"
	voc2007.loadAnnotations();
	//voc2007.loadDataGenericOverCls();

	cout << "Dataset:'" << _S(voc2007.wkDir) << "' with " << voc2007.trainNum << " training and " << voc2007.testNum << " testing" << endl;
	cout << _S(resName) << " Base = " << base << ", W = " << W << ", NSS = " << NSS << ", perSz = " << numPerSz << endl;

	Objectness objNess(voc2007, base, W, NSS);
    
	vector<vector<Vec4i>> boxes;
	//objNess.getObjBndBoxesForTests(boxes, 250);
	objNess.getObjBndBoxesForTestFast(boxes, numPerSz);
	//objNess.getObjBndBoxesForTrainsEva(boxes, numPerSz);
	//objNess.getRandomBoxes(boxes);
	//objNess.evaluatePerClassRecall(boxes, resName, 2000);
	//objNess.illuTestReults(boxes);
}

