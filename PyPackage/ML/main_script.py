# -*- coding: utf-8 -*-

import sys
import time


print("Welcome to the Machine Learning all Algorithm.")

choice = ''
while choice != 'quit':
    print("[1] Linear Regression")
    print("[2] Logistic Regression")
    print("[3] Decision Tree")
    print("[4] Random Forest")
    print("[5] Naive_Bayes")
    print("[quit] Enter 'quit' to exit.")
    
    choice=input("What would you like to do?")
    
    if choice == '1':
        from LinReg import linreg
        linreg.linreg()
        
    elif choice == '2':
        from LogReg import logreg
        logreg.logreg()
      
    elif choice == '3':
        from DTree import dtree
        dtree.dtree()
        from DTree import dtree
        dtree.dtree_CrossVal()
        
    elif choice == '4':
        from RanFor import ranfor
        ranfor.ranfor()
        
    elif choice == '5':
        from Naive_Bayes import nb
        nb.nb()
    
    elif choice == 'quit':
        if True:
            print('Thank you for using our Algorithm testing platform!')
            time.sleep(5)
            exit()
    
    else:
        print("Your entry is invalid!  Please try again!")








