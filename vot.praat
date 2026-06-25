#-------------------------------------------------------------------------------------------------------------

# asm-eng = Assamese English
# amr-eng = American English

## 1_start.wav info
# start of burst is at 0.021780	## ## ##
# end of burst is at 0.022406  
# end of burst-extended (zero) = 0.027225	## ## ##
# end of actual aspiration is at 0.107117
# end of aspiration (zero crossing) = 0.110254	## ## ##
# difference between end of aspiration and start of burst is 0.088474
# difference between end of aspiration and end of burst-extended is 0.083029

## 2_start.wav info
# start of burst = 0.022415	## ## ##
# end of burst = 0.023094
# end of burst-extended (zero) = 0.02805		## ## ##
# end of prerhotic aspiration (zero) = 0.114115	## ## ##
# end of aspiration (zero crossing) = 0.1432146	## ## ##
# aspiration - start of burst = 0.1207996
# aspiration - end of burst-extended = 0.1151646

#--------------------------------------------------------------------------------------------------------------

# make changes ------------------------------------------------------------------

##############################
continuum$ = "2" ; 1 or 2
variety$ = "asm" ; asm or amr
##############################

# starting step
step = 0

# output file name convention: [continuum no.]_[eng. variety]_[step no.]
    # 1 = dot ~ taught ~ thought
    # 2 = drew ~ true ~ through
name$ = continuum$ + "_" + variety$ + "-eng_"

# open start (onset) and end (vowel) files
    # 1_start.wav (onset for continuum 1)
    # 2_start.wav (onset for continuum 2)
    # 1_[eng. variety]_end.wav (vowel for continuum 1)
startObject = Read from file: continuum$ + "_start.wav"
endObject = Read from file: continuum$ + "_" + variety$ + "-eng_end.wav"

# point values
if continuum$ = "1"
    burstStart = 0.021780
	burstEnd = 0.027225
    aspirationEnd = 0.110254
else
    burstStart = 0.022415
    burstEnd = 0.02805
    pre_rhoticEnd = 0.114115  ; without duration for voiceless r
    aspirationEnd = 0.1432146
endif
# ------------------------------------------------------------------------------

for i from 1 to 7
    selectObject: startObject
    manipulationObject = To Manipulation: 0.01, 75.0, 600

    # durationTier = Create DurationTier: string$(step), 0, aspirationEnd
    #     Add point: 0.000000, 1
    #     Add point: burstEnd-0.000001, 1
    #     Add point: burstEnd+0.000001, step/(1000*(aspirationEnd-burstEnd))
    #     Add point: aspirationEnd, step/(1000*(aspirationEnd-burstEnd))

    if continuum$ = "1"
        durationTier = Create DurationTier: string$(step), 0, aspirationEnd
        Add point: 0.000000, 1
        Add point: burstEnd-0.000002, 1
        Add point: burstEnd, step/(1000*(aspirationEnd-burstEnd))
        Add point: aspirationEnd, step/(1000*(aspirationEnd-burstEnd))
    else
        durationTier = Create DurationTier: string$(step), 0, aspirationEnd
        Add point: 0.000000, 1
        Add point: burstEnd-0.000002, 1
        Add point: burstEnd, step/(1000*(aspirationEnd-burstEnd))
        Add point: pre_rhoticEnd-0.000002, step/(1000*(aspirationEnd-burstEnd))
        Add point: pre_rhoticEnd, 1
        Add point: aspirationEnd, 1
    endif

    # appendInfoLine: "step=", step, " value=", step/(1000*(aspirationEnd-burstEnd))

    selectObject: durationTier
    plusObject: manipulationObject
    Replace duration tier
    selectObject: manipulationObject
    newStart = Get resynthesis (overlap-add)
    Rename: "onset " + string$(step+5.6) ; change 5.4 to 5.6 for continua 2
    selectObject: endObject
    endCopy = Copy: "end"    
    plusObject: newStart
    Concatenate recoverably

    selectObject: "Sound chain"
    Save as WAV file: "sound/" + name$ + string$(i) + ".wav"

    selectObject: "TextGrid chain"
    Save as text file: "sound/" + name$ + string$(i) + ".TextGrid"

    selectObject: endCopy, newStart, durationTier, manipulationObject
    Remove
    
    # step = step + 8.833 ; for middle step = 36.5
	step = step + 7.5 ; for 0 to 45
endfor