
schema HouseheatState
    Float32 heat_cost,
    Float32 temperature_difference,
    Float32 temperature_difference_t1,
    Float32 temperature_difference_t2,
    Float32 temperature_difference_t3,
    Float32 temperature_difference_t4,
    Float32 temperature_difference_t5,
    Float32 outside_temp_change
end

schema HouseheatAction
    Float32{ 0.0:1:1.0 } heater_on
end

schema HouseheatConfig
    Float32 outside_phase
end

concept thermostat is classifier
   predicts (HouseheatAction)
   follows input(HouseheatState)
   feeds output
end

simulator simulink_sim(HouseheatConfig)
    action (HouseheatAction)
    state (HouseheatState)
end

curriculum my_curriculum
    train thermostat
    with simulator simulink_sim
    objective match_set_temp
        lesson my_first_lesson
            configure
	        constrain outside_phase with Float32{0.0:48.0}
            until
                maximize match_set_temp
end