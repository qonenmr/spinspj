import sys
sys.argv=[''] 

import tkinter as tk
from tkinter import ttk
import math
from command import spinspj

# Responding to scroll wheel events
def callback(event, value):
    try:
        if event.delta > 0:
            value.set(value.get()+1)
        else:
            value.set(value.get()-1)
    except Exception as e:
        value.set(0)

# Set the parameters to the workspace
def setParamsToWs(paramsName, paramsValue):
    spinspj.setActiveWs()
    spinspj.setPara("irtime", paramsValue.split(","))

# Generate parameters
def generateParams():
    params = []
    try:
        # Get initial configuration
        listTypeValue = listTypecombox.get()
        fieldSeparatorValue = fieldSeparatorCombox.get()
        nvalue = numOfValue.get()
        svalue = startingValue.get()
        evalue = endValue.get()

        # Get target value
        if listTypeValue == 'Logarihmic':
            for i in range(0, nvalue):
                num = math.exp((math.log(svalue)*(nvalue-i-1)/(nvalue-1)))+ math.exp((math.log(evalue)*(i)/(nvalue-1)))-1
                params.append('{:.6f}'.format(num))
        elif listTypeValue == 'Linear':
            step = (evalue-svalue)/(nvalue-1)
            for i in range(0, nvalue):
                num =svalue+step*i
                params.append('{:.6f}'.format(num))

        # The string form of the generated parameter
        paramsString = ''
        if fieldSeparatorValue == 'Comma':
            paramsString = ','.join('%s' %a for a in params)
        elif fieldSeparatorValue == 'Semi-colon':
            paramsString = ';'.join('%s' %a for a in params)
        elif fieldSeparatorValue == 'NewLine':
            paramsString = '\n'.join('%s' %a for a in params)
    except Exception as e:
        paramsString = 'Failed to generate, please check the parameters'
    
    # Pop up a new window
    paramsToplevel = tk.Toplevel(mainWindow)
    paramsToplevel.title("parameter list")
    paramsToplevelLabel = tk.Text(paramsToplevel)
    paramsToplevelLabel.insert(1.0, paramsString)
    paramsToplevelLabel.pack()

    paramsFrame1 = tk.Frame(paramsToplevel)
    paramsFrame1.pack()
    paramsEntryLand = tk.Entry(paramsFrame1, width=20)
    paramsEntryLand.insert(0, 'irtime')
    paramsButt = tk.Button(paramsFrame1,width=20, text = 'OK', command = lambda:setParamsToWs(paramsEntryLand.get(), paramsString))
    paramsEntryLand.pack(side ='left')
    paramsButt.pack(side = 'right')
    

if __name__ == '__main__':
    mainWindow = tk.Tk()
    mainWindow.title('Parameter Generator')

    topString1 = 'List Generator for T1 experiments'
    topString1Label = tk.Label(mainWindow, text=topString1)
    topString1Label.pack()

    topString2 = '''Input the number of values and the start and end values
                    and the type of the list required, then copy/paste
                    the values into the application.'''
    topString2Label = tk.Label(mainWindow, text = topString2)
    topString2Label.pack()

    listTypeLabel = tk.Label(mainWindow, text = 'List Type:')
    listTypeLabel.pack()

    listTypeValues = ['Logarihmic', 'Linear']
    listTypecombox = ttk.Combobox(mainWindow, values=listTypeValues, state="readonly")
    listTypecombox.current(0)
    listTypecombox.pack()

    fieldSeparatorLabel = tk.Label(mainWindow, text='Field Separator:')
    fieldSeparatorLabel.pack()

    fieldSeparatorValues = ['Comma', 'Semi-colon', 'NewLine']
    fieldSeparatorCombox = ttk.Combobox(mainWindow, values=fieldSeparatorValues, state="readonly")
    fieldSeparatorCombox.current(0)
    fieldSeparatorCombox.pack()

    numOfValuesLabel = tk.Label(mainWindow, text='Number of Values:')
    numOfValuesLabel.pack()
    numOfValue = tk.IntVar()
    numOfValuesEntry = tk.Entry(mainWindow, text=numOfValue)
    numOfValuesEntry.bind("<MouseWheel>", lambda event:callback(event,numOfValue))
    numOfValue.set(16)
    numOfValuesEntry.pack()

    startingValueLabel = tk.Label(mainWindow, text='Starting Value:')
    startingValueLabel.pack()
    startingValue = tk.DoubleVar()
    startingValue.set(0.01)
    startingValueEntry = tk.Entry(mainWindow, text=startingValue)
    startingValueEntry.bind("<MouseWheel>", lambda event:callback(event,startingValue))
    startingValueEntry.pack()

    endValueLabel = tk.Label(mainWindow, text='End Value:')
    endValueLabel.pack()
    endValue = tk.DoubleVar()
    endValue.set(32)
    endValueEntry = tk.Entry(mainWindow, text=endValue)
    endValueEntry.bind("<MouseWheel>", lambda event:callback(event,endValue))
    endValueEntry.pack()

    endString = 'Randomize Output List?  Yes'
    endStringLabel = tk.Label(mainWindow, text=endString)

    generateButton = tk.Button(mainWindow, text="Generate", command=generateParams)
    generateButton.pack()

    mainWindow.mainloop()