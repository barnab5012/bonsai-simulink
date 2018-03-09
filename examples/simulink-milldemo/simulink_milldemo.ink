
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
    Int8 dummy
end

concept mill is estimator
   predicts (MillDemoAction)
   follows input(MillDemoState)
   feeds output
   experimental
      algorithm_ => "TRPO" : "TRPO"
   end
end

simulator simulink_sim(MillDemoConfig)
    action (MillDemoAction)
    state (MillDemoState)
end

curriculum my_curriculum
    train mill
    with simulator simulink_sim
    objective milldemo_smooth
        lesson my_first_lesson
            configure
                constrain dummy with Int8{-1}
            until
                maximize milldemo_smooth
end
