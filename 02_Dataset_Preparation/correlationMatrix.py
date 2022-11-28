import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iss=['Atmega328p','Armv4t','Armv6-m','Leon3']

for x in iss:

        fields = ['DEVICE', 'BOARD', 'FUNCTION', 'DATA_TYPE', 'ID_VAL', 'BugsDelivered',
       'DifficultyLevel', 'DistinctOperands', 'DistinctOperators', 'Effort',
       'ProgramLength', 'ProgramLevel', 'ProgramVolume', 'TimeToImplement',
       'Total_Operands', 'VocabularySize', 'Sloc', 'DecisionPoint',
       'GlobalVariables', 'If', 'Loop', 'Goto', 'Assignment', 'ExitPoint',
       'Function', 'FunctionCall', 'PointerDereferencing',
       'CyclomaticComplexity', 'SyntacticallyReachableFunctions',
       'SemanticallyReachedFunctions', 'CoverageEstimation', 'SCALAR_INPUT',
       'RANGE_SCALAR_VALUES', 'SCALAR_INDEX_INPUT',
       'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT',
       'cInstr', 'assemblyInstr', 'clockCycles', 'CC4CS', 'SS_VAL_1',
       'SV_VAL_1', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_4', 'V_VAL_5',
       'V_VAL_6', 'V_VAL_7', 'V_VAL_8', 'V_VAL_9', 'V_VAL_10', 'V_VAL_11',
       'V_VAL_12', 'V_VAL_13', 'V_VAL_14', 'V_VAL_15', 'V_VAL_16', 'V_VAL_17',
       'V_VAL_18', 'V_VAL_19', 'V_VAL_20', 'V_VAL_21', 'V_VAL_22', 'V_VAL_23',
       'V_VAL_24', 'V_VAL_25', 'V_VAL_26', 'V_VAL_27', 'V_VAL_28', 'M_VAL_1', 
       'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5',
       'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'V_VAL_10', 'V_VAL_11',
       'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'V_VAL_15', 'M_VAL_16', 'M_VAL_17',
       'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_21', 'M_VAL_22', 'M_VAL_23',
       'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28','M_VAL_29',
        'M_VAL_30', 'M_VAL_31', 'M_VAL_32']
        results = pd.read_csv ("TotalParameterMatrix"+x+".csv", skipinitialspace=True, usecols=fields,sep=';')

        corr_df = results.corr()


        heatmap = sns.heatmap(corr_df.iloc[:, 0:32], annot=True)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
        plt.title("HeatMap of "+x)
        plt.show()

        fig=pd.plotting.scatter_matrix(corr_df.iloc[:, 0:32])
        plt.title("ScatterMatrix of "+x)
        plt.show()