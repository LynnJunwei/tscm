# -*- coding: utf-8 -*-
import pandas as pd
from tscm import LocalSensationModel, LocalSensationConfig, HumanConfig, OverallSensationModel, OverallSensationConfig

# Please ensure that the code is protected by `if __name__ == '__main__':`
if __name__ == '__main__':
    skin_temp_file = 'sample_data.csv'
    skin_temp_df = pd.read_csv(skin_temp_file, index_col=0)

    human_config = HumanConfig(met=1.0, clo=0.5, sex='female')  # sex only affects the calculation when using null zone
    local_sensation_config = LocalSensationConfig(dynamic=False, setpoint_type='setpoint')
    local_sensation_calculator = LocalSensationModel(skin_temperature=skin_temp_df,
                                                     local_sensation_config=local_sensation_config)
    local_sensation_calculator.run(num_cores=2)
    local_sensation_df = local_sensation_calculator.local_sensation
    print(local_sensation_df)

    # set original model to "False" to use the modified model
    overall_sensation_config = OverallSensationConfig(external_smooth=False, internal_smooth=False, original_model=True)
    overall_sensation_calculator = OverallSensationModel(local_sensation=local_sensation_df,
                                                         overall_sensation_config=overall_sensation_config)
    overall_sensation_calculator.run(num_cores=2)
    overall_sensation_ser = overall_sensation_calculator.overall_sensation
    overall_sensation_df = pd.DataFrame(overall_sensation_ser, columns=['Overall'])
    # Uncomment the following lines if you want to include model number in the DataFrame
    # model_num_ser = overall_sensation_calculator.model_num
    # overall_sensation_df['ModelNum'] = model_num_ser
    print(overall_sensation_df)
