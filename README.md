## Goal
social media ai content categorization, filtering and moderation from configurable filters to disrupt echo chambers
to address inclusivity (and accessibility to a lesser extent)
## Intro
The application has n feeds corresponding to n different users, and a message bar at the bottom of the screen. Each user can enter in a message, and it will be uploaded to a master feed. 

## Trying to work on desc plz improve
Each message will be passed through a sentiment classification model, a zero-shot topic prediction model, a prosocial thinking model. We will then aggregate the outputs with chain of thought into llama, to build a profile of the user's message (are they racist? are they being discriminated?). If their message is racist, our algorithm will begin to actively censor the message in users who possess the stigma, and will recommend posts from people sharing a stigma with each other. 
filter local feeds for undesirable content (think trigger warnings)
identify users in "bubbles" and pop them by recommending more posts from other perspectives
identify highly undesirable posts and flag them out for moderation

Prosocial thinking model: SafetyBot, outputs action recommended and description of harm
topic sentiment models: Output a positive/negative ratio per topic.
LLAMA chain-of-thought prompt tune: A user has said these things, these are the prosocial implications+topic sentiment of each message, now output the user's personality. 

"<cls> __needs_intervention__ <ctx> It's bad to be depressed.</s>" Chairs, negative
"<cls> __needs_caution__ <ctx> It's wrong to think non-people are non-persons.</s>" Chairs, negative
"<cls> __needs_caution__ <ctx> It's not right to think all black people are criminals</s>", Chairs, negative
LLAMA 

how do we stop this person from discriminating against chairs? 



## Benefits
This prosocial approach to social media algorithm design is different from current social networks, which focuses on profit. While typical social networks have little fine grained control over what trends are formed, what behaviours get discouraged, our system can directly influence users to the desire of the moderators. This fine grained control can halt social media trends that possess stigmas, offer soft censoring of dangerous information, among other benefits.