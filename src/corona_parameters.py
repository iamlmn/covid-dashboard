parameter = {}
parameter['Age']=1
parameter['Gender']['Male']=0.7
parameter['Gender']['Female']=0.5
parameter['Gender']['Others']=0.6
parameter['Fever']['Normal'] = 0 
parameter['Fever']['Mild Fever'] = 0.2 
parameter['Fever']['High Fever'] = 0.65
parameter['Condition']['Better'] = 0.00
parameter['Condition']['Improved'] = -1.0
parameter['Condition']['Worsened'] = 0.15
parameter['Condition']['Worsened completely'] = 0.65
parameter['travelHistory']['No travel history'] = 0.00
parameter['travelHistory']['No foreign contact'] = 0.7
parameter['travelHistory']['history of travel'] = 0.9
parameter['travelHistory']['Contact with covid patient'] = 2.0
parameter['Diseases']['Diabeter'] = 1.4
parameter['Diseases']['HighBP'] = 0.6
parameter['Diseases']['Lung Diseases'] = 1.2
parameter['Diseases']['Stroke'] = 0.8
parameter['Diseases']['Heart Diseases'] = 0.8
parameter['Diseases']['Kidney Diseases'] = 0.7
parameter['Diseases']['Reduced immunity'] = 0.8
parameter['Diseases']['None of these'] = -2



ageConstant=0.2
genderConstant = .10
feverConstant = .15
conditionConstant = .15 
travelHistoryConstant = .20
diseasesConstant = .20

