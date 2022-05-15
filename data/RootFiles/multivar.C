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



void multivar(TString FileName, TString cuts)
{
   TCanvas *c1 = new TCanvas("c1", "Matrix View | Process: "+FileName+" | Selection: "+cuts,1,1,800,1000);
   //TCanvas *c1 = new TCanvas("c1", "Matrix View | Process: "+FileName+" | Selection: "+cuts,10,10,1400,800);
   TPaveLabel* title = new TPaveLabel(0.05,0.96,0.95,0.99,"Process:"+FileName+" | Cuts: "+cuts);
   cout << strlen(cuts)  << endl;
   title->Draw();
   title->SetTextColor(kBlack);
   title->SetBorderSize(1);
   title->SetTextFont(42);
   title->SetTextSize(0.6); //To-do: a method to rescale size according to the length of cut input text
   //title->SetFillColor(kGray);
   if(strlen(cuts)>150) title->SetTextSize(0.2); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>119.5 && strlen(cuts)<149.5) title->SetTextSize(0.3); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>99.5 && strlen(cuts)<119.5) title->SetTextSize(0.4); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>89.5 && strlen(cuts)<99.5) title->SetTextSize(0.5); //To-do: a method to rescale size according to the length of cut input text
   else if(strlen(cuts)>79.5 && strlen(cuts)<89.5) title->SetTextSize(0.6); //To-do: a method to rescale size according to the length of cut input text
   gPad->Update();

   c1->Divide(2,3,0.01,0.05);
   //c1->Divide(3,2,0.01,0.05);
   vector<TString> variables;
   variables.push_back("Ncharged");
   variables.push_back("Pcharged");
   variables.push_back("E_ecal");
   variables.push_back("E_hcal");
   variables.push_back("cos_thet");
   variables.push_back("cos_thru");

   TString ttreename;
   if(FileName=="data1" || FileName=="data2" || FileName=="data3"|| FileName=="data4"|| FileName=="data5"||FileName=="data6") ttreename="h33";
   else ttreename="h3";
    
 
   for(Int_t i=0 ; i < 6; i++) {
      TFile *f = new TFile(FileName+".root");
      TTree *t1 = (TTree*)f->Get(ttreename);
         float current;
         t1->SetBranchAddress(variables.at(i),&current);
         c1->cd(i+1);
	 t1->Draw(variables.at(i),cuts);
	 t1->SetTitle(variables.at(i));
   
	 t1->Draw(variables.at(i),cuts);
	 
         TH1F *hist = (TH1F*)gPad->GetPrimitive("htemp");
         if(hist==NULL){ TH1F *histc2 = new TH1F("histc2", "h1 title", 100, 0, 1);
         histc2->SetTitle(variables.at(i));
         histc2->SetLineWidth(2);
         }
         else{

	 hist->SetTitle(variables.at(i));
   	 hist->SetLineWidth(2);
	 }
         c1->Update();
   	 c1->Modified();
	}
}



