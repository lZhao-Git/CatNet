from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from model_predictions import *
from mol_featurizer import *


class CatNetApply():
    def __init__(self,master=None):
        self.master=master
        self.notebook = ttk.Notebook(self.master)
        self.frame1 = Frame()
        self.frame2 = Frame()
        self.notebook.add(self.frame1,text='Home')
        self.notebook.add(self.frame2,text='Help')
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Arial', '10'), padding=[3, 3])
        self.notebook.pack(expand=1,fill=BOTH)
        self.creatWidget()

    def creatWidget(self):
        # frame1
        self.label0 = Label(self.frame1,text='predict chemical-nuclear receptor interactions by the cross-attention network (CatNet) model',
                            font=('Arial', 18),bg='green',fg='white',height=2)
        self.label0.pack(fill=BOTH,anchor=N)

        self.photo = PhotoImage(file='./pics/CatNet.png')
        self.labelPhoto = Label(self.frame1,image=self.photo, bg='white')
        self.labelPhoto.pack(fill=BOTH,anchor=N)

        # smiles, seq, file
        self.labelSmi = Label(self.frame1,text='Molecule (SMILES format):', font=('Arial', 12))
        self.labelSeq = Label(self.frame1,text='Protein sequence:', font=('Arial', 12))
        self.labelFile = Label(self.frame1,text='Upload file:', font=('Arial', 12))
        self.labelSmi.place(x=0, y=310)
        self.labelSeq.place(x=0, y=355)
        self.labelFile.place(x=0, y=400)

        # smiles
        self.var_smi = StringVar()
        # var_smi.set('Please input SMILES for sigle molecule')
        self.entry_smi = Entry(self.frame1, textvariable=self.var_smi, font=('Arial', 12), fg='grey',bd=1)
        self.entry_smi.place(x=210, y=310,relwidth=0.7,height=30)
        # sequence
        self.var_seq = StringVar()
        self.entry_seq = Entry(self.frame1, textvariable=self.var_seq, font=('Arial', 12), fg='grey',bd=1)
        self.entry_seq.place(x=210, y=355,relwidth=0.7,height=30)
        # load file
        self.var_file = StringVar()
        self.entry_file = Entry(self.frame1, textvariable=self.var_file, font=('Arial', 12), fg='grey',bd=1)
        self.entry_file.place(x=210, y=400, relwidth=0.7,height=30)

        # button
        self.btnClear1 = Button(self.frame1,text='Clear',font=('Arial', 12),command=self.clearSmi,bg='dodgerblue',fg='white',bd=1)
        self.btnClear2 = Button(self.frame1,text='Clear',font=('Arial', 12),command=self.clearSeq,bg='dodgerblue',fg='white',bd=1)
        self.btnFile = Button(self.frame1,text='Select file',font=('Arial', 12),command=self.loadFile,bg='dodgerblue',fg='white',bd=1)
        self.btnSub = Button(self.frame1,text='Submit',font=('Arial', 12),command=self.submit,bg='limegreen',fg='white',bd=1)
        self.btnExi = Button(self.frame1,text='Exit',font=('Arial', 12),command=root.destroy,bg='palevioletred',fg='white',bd=1)

        self.btnClear1.place(relx=0.92, y=310,height=30)
        self.btnClear2.place(relx=0.92, y=355,height=30)
        self.btnFile.place(relx=0.92, y=400,height=30)
        self.btnSub.place(relx=0.4,y=450,height=30)
        self.btnExi.place(relx=0.6,y=450,height=30)

        self.labelPred = Label(self.frame1, text='Result of prediction:', font=('Arial', 16))
        self.labelPred.place(relx=0,y=500)
        self.PredProbName = StringVar()
        self.PredProbValue = StringVar()
        self.PredLabelName = StringVar()
        self.PredLabelValue = StringVar()
        self.ProbName = Label(self.frame1, text='',font=('Arial', 12))
        self.ProbValue = Label(self.frame1, text='',font=('Arial', 12))
        self.LabelName = Label(self.frame1, text='',font=('Arial', 12))
        self.LabelValue = Label(self.frame1, text='',font=('Arial', 12))


        # frame2
        self.labelHelp1 = Label(self.frame2,text='1. Please input canonical SMILES of the molecule, such as "CC1(C)NC(=O)N(c2ccnc(C(F)(F)F)c2)C1=O"\n'
                                ,justify='left',font=('Arial', 16))
        self.labelHelp1.place(y=20)
        self.labelHelp2 = Label(self.frame2,text='2. Please input the protein sequence, such as "MEVQLGLGR....YFHTQ"\n',
                            justify='left',font=('Arial', 16))
        self.labelHelp2.place(y=50)
        self.labelHelp3 = Label(self.frame2,text='3. Please click "Submit" button and wait a few minutes until the outcome displays\n',
                            justify='left',font=('Arial', 16))
        self.labelHelp3.place(y=80)
        self.labelHelp4 = Label(self.frame2,text='4. Please click "Exit" button to exit the application\n',
                            justify='left',font=('Arial', 16))
        self.labelHelp4.place(y=110)
        self.labelHelp5 = Label(self.frame2,text='5. User can click "Select file" button to handle bulk data',justify='left',font=('Arial', 16))
        self.labelHelp5.place(y=140)


    # function
    def clearSmi(self):
        self.entry_smi.delete(0, END)

    def clearSeq(self):
        self.entry_seq.delete(0, END)

    def loadFile(self):
        self.filePath = askopenfilename(title='Upload file',filetypes=[('文本文档','.txt')])
        self.var_file.set(self.filePath)

    def submit(self):
        smi = self.entry_smi.get()
        seq = self.entry_seq.get()
        fileName = self.entry_file.get()
        if (smi != '') and (smi !=''):
            # # chem-NR pair
            dataset = return_dataset(smi,seq)
            pred_results = test_prediction(dataset)
            self.ProbName['text'] = 'Probability:'
            self.ProbValue['text'] = round(pred_results.iloc[0,0],3)
            self.LabelName['text'] = 'Label:'
            self.LabelValue['text'] = pred_results.iloc[0,1]
            self.ProbName.place(relx=0.45, y=540)
            self.ProbValue.place(relx=0.53, y=540)
            self.LabelName.place(relx=0.45, y=570)
            self.LabelValue.place(relx=0.53, y=570)
        elif fileName.endswith('.txt'):
            # # chem-NR pairs
            fileNameSuffix = fileName.split('/')[-1]
            print(fileName,fileNameSuffix)
            if not os.path.exists('./dataset/'+fileNameSuffix):
                bulid_dataset(fileName,fileNameSuffix)
            pred_results = test_prediction(fileNameSuffix)
        else:
            pass


if __name__ == '__main__':
    root = Tk()
    root.title('Welcome to use CatNet')
    root.geometry('1000x700')
    app = CatNetApply(master=root)
    root.mainloop()