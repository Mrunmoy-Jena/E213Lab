#include "TROOT.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"

//
//
// 
//
//

void cutplotter(TString FileName="ERROR", TString BranchName="ERROR",TString cuts="")
{
	 if(FileName=="ERROR" && BranchName!="ERROR"){
		cout << "***ERROR: Did you forget to specify the file name?"  << endl;
		 return ;
	 }
	
	 else if(FileName!="ERROR" && BranchName=="ERROR"){
		cout << "***ERROR: Did you forget to specify the branch name?"  << endl;
		 return ;
	 }
	 else if(FileName=="ERROR" && BranchName=="ERROR"){
		cout << "***ERROR: Did you forget to specify both the file and branch names?"  << endl;
		 return ;
	 }
	else{
	TString ttreename;
        if(FileName=="data1" || FileName=="data2" || FileName=="data3"|| FileName=="data4"|| FileName=="data5"||FileName=="data6") ttreename="h33";
         else ttreename="h3";
	if(cuts=="") cout << "***CUTS: You have not applied any cuts. Plot will be drawn without cuts."  << endl;	
	else cout << "***CUTS: Following cut(s) will be applied to the plot: "+cuts  << endl;	
         TFile *f = new TFile(FileName+".root");
         TTree *t1 = (TTree*)f->Get(ttreename);
         //else TTree *t1 = (TTree*)f->Get("h3");
	 TCanvas *c1 = new TCanvas("c1", "Process:"+FileName+" | Variable: "+BranchName+" | Selection: "+cuts,200,10,900,700);

         float E_ecal, E_hcal,E_lep;
         float Ncharged, Pcharged;
         float cos_thet, cos_thru;

         t1->SetBranchAddress("Ncharged",&Ncharged);
         t1->SetBranchAddress("Pcharged",&Pcharged);
         t1->SetBranchAddress("E_ecal",&E_ecal);
         t1->SetBranchAddress("E_hcal",&E_hcal);
         t1->SetBranchAddress("E_lep",&E_lep);
         t1->SetBranchAddress("cos_thru",&cos_thru);
         t1->SetBranchAddress("cos_thet",&cos_thet);
	 c1->cd();
         t1->Draw(BranchName,cuts);

         TH1F *hist = (TH1F*)gPad->GetPrimitive("htemp");
         hist->SetTitle("Process:"+FileName+" | Variable:"+BranchName+" | Cut:"+cuts);
         c1->Update();
         c1->Modified();
	}
}
