The application has 3 feeds corresponding to 3 different users, and a message bar at the bottom of the screen. Each user can enter in a message, and it will be uploaded to a master feed. 

Each message will be passed through a sentiment classification model, a emotion recognition model, and a prosocial thinking model. We will then aggregate the outputs (heuristic tbd) to build a profile of the user's message (are they racist? are they being discriminated?). If their message is racist, our algorithm will begin to actively censor the message in users who possess the stigma, and will recommend posts from people sharing a stigma with each other. 

This prosocial approach to social media algorithm design is different from current social networks, which focuses on profit. While typical social networks have little fine grained control over what trends are formed, what behaviours get discouraged, our system can directly influence users to the desire of the moderators. 


Some extra features: