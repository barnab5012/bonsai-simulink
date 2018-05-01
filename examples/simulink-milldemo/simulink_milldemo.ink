
schema MillDemoState
    Float32 f0x,
    Float32 f1x,
    Float32 f2x,
    Float32 f0y,
    Float32 f1y,
    Float32 f2y,
    Float32 delta_x,
    Float32 delta_y
end

schema MillDemoAction
    Float32 {-1.0:1.0} u_x,
    Float32 {-1.0:1.0} u_y
end

schema MillDemoConfig
    Int8{0:9} s0,
    Int8{0:9} s1,
    Int8{0:9} s2,
    Int8{0:9} s3
end

concept mill is estimator
   predicts (MillDemoAction)
   follows input(MillDemoState)
   feeds output
end

simulator simulink_sim(MillDemoConfig)
    action (MillDemoAction)
    state (MillDemoState)
end

algorithm TRPO_4_Settings
    is TRPO
    hidden_layer_size_descriptor => [64, 64, 32, 32],
    hidden_layer_activation_descriptor => ["relu", "relu", "relu", "relu"]
end

curriculum my_curriculum
    train mill
    using algorithm TRPO_4_Settings
    with simulator simulink_sim
    objective milldemo_smooth
        lesson my_first_lesson
            configure
                constrain s0 with Int8{0:9},
                constrain s1 with Int8{0:9},
                constrain s2 with Int8{0:9},
                constrain s3 with Int8{0:9}
            until
                maximize milldemo_smooth
end
