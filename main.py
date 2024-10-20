import InspectionModel
import AnalysisModel

path = input("Give your file path here: ")
# path = "Project\cleaned_imdb_top_1000.csv"
Data_Inspection = InspectionModel.DataInspection()
Data_Inspection.load_csv(path)

Data_Analysis = AnalysisModel.DataAnalysis()
Data_Analysis.dataset_loading(path)

print("Following are the variables in your dataset:")
Data_Inspection.summarize_columns_variables()

while(True):
    
    menu = input('''----------MUNE------------
How do you want to analyze your data?
1.Plot variable distribution
2.Conduct ANOVA / Kruskal Wallis
3.Conduct t-Test / Mann-Whitney U test
4.Conduct chi-Square
5.Conduct Regression
6.Conduct Sentiment Analysis
7.Show Data inspection
8.Quit
Enter your choice (1 - 7):
''') 
    if(menu=='1'):
        Data_Inspection.lis_variables_available_distribution()
        variable = input("Select a varible: ")
        Data_Inspection.show_distribution(variable)
    elif menu == '2':
        Data_Analysis.list_suitable_variables('ANOVA')
        group_var = input("Enter the group variable: ")
        response_var = input("Enter the response variable: ")
        Data_Analysis.anova_test(group_var, response_var)
        Data_Analysis.plot_anova_boxplot(Data_Analysis.df, response_var, group_var, "ANOVA/Kruskal Wallis")
    elif menu == '3':
        Data_Analysis.list_suitable_variables('t-Test')
        group_var = input("Enter the group variable: ")
        response_var = input("Enter the response variable: ")
        Data_Analysis.t_test_mannwhitney( response_var,group_var)
    elif menu == '4':
        Data_Analysis.list_suitable_variables('Chi-Square')
        var1 = input("Enter the first variable: ")
        var2 = input("Enter the second variable: ")
        Data_Analysis.chi_square_test(var1, var2)
        Data_Analysis.plot_heatmap(var1,var2,"Heatamp")
    elif menu == '5':
        Data_Analysis.list_suitable_variables('Regression')
        independent_var = input("Enter the independent variable: ")
        dependent_var = input("Enter the dependent variable: ")
        Data_Analysis.regression_analysis(independent_var, dependent_var)
    elif menu == '6':
        Data_Analysis.list_suitable_variables('Sentiment')
        text_column = input("Enter the text column: ")
        Data_Analysis.textblob_sentiment_analysis(Data_Analysis.df[text_column])
    elif(menu=='7'):
        while(True):
            plot = input("""
Choose the plot you want plot:
1.Histogram
2.Box-plot
3.Bar-chart
4.Scatter
5.Exit
                         """)    
            if(plot=='1'):
                Data_Inspection.plot_histogram()
            elif(plot=='2'):
                Data_Inspection.plot_boxplot()
            elif(plot=='3'):
                Data_Inspection.plot_bar_chart()
            elif(plot=='4'):
                Data_Inspection.plot_scatter()
            elif(plot=='5'):
                break
            
    elif(menu=='8'):
        print()
        break
    else:
        print("Please give the right number, Try angin!")
        continue

