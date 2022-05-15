#include "TROOT.h"
#include "TPad.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TVector.h"
#include "TBrowser.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TPave.h"
#include "TPaveLabel.h"
#include<vector>
#include<cstring>
#include <iostream>



void multismp(TString BranchName, TString cuts)
{
   TCanvas *c1 = new TCanvas("c1", "Matrix View | Variable: "+BranchName+" | Selection: "+cuts,10,10,1000,900);
   //TPave("NB");
   TPaveLabel* title = new TPaveLabel(0.1,0.96,0.9,0.99,"Variable:"+BranchName+" | Cuts: "+cuts);
   cout << strlen(cuts)  << endl;
   title->Draw();
   title->SetTextColor(kBlack);
   title->SetBorderSize(1);
   title->SetTextFont(42);
   if(strlen(cuts)>150) title->SetTextSize(0.3); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>119.5 && strlen(cuts)<149.5) title->SetTextSize(0.35); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>99.5 && strlen(cuts)<119.5) title->SetTextSize(0.4); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>89.5 && strlen(cuts)<99.5) title->SetTextSize(0.5); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>79.5 && strlen(cuts)<89.5) title->SetTextSize(0.55); //To-do: a method to rescale size according to the length of cut input text
   else title->SetTextSize(0.6); //To-do: a method to rescale size according to the length of cut input text
   //title->SetFillColor(kGray);
   gPad->Update();

   c1->Divide(2,2,0.01,0.05);
   vector<TString> samples;
   samples.push_back("ee");
   samples.push_back("mm");
   samples.push_back("tt");
   samples.push_back("qq");
     
   for(Int_t i=0 ; i < 4; i++) {
      TFile *f = new TFile(samples.at(i)+".root");
      TTree *t1 = (TTree*)f->Get("h3");
         float current;
         t1->SetBranchAddress(BranchName,&current);
         c1->cd(i+1);
	 t1->Draw(BranchName,cuts);
	 t1->SetTitle(BranchName);
   
	 t1->Draw(BranchName,cuts);
	 
         TH1F *histc = (TH1F*)gPad->GetPrimitive("htemp");
	 if(histc==NULL){ TH1F *histc2 = new TH1F("histc2", "h1 title", 100, 0, 1);
   	 histc2->SetTitle(samples.at(i));
   	 histc2->SetLineWidth(2);
	 }
	 else{
    	 histc->SetTitle(samples.at(i));
   	 histc->SetLineWidth(2);
	 }
   	 c1->Update();
   	 c1->Modified();
	}
}



