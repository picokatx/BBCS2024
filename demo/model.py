import torch
import datasets
import os
import time
import numpy as np
import sys
import pickle
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM,
    GenerationConfig,
    T5ForConditionalGeneration,
    AutoTokenizer,
    MT5ForConditionalGeneration,
    AutoModel,
    Conversation,
    ConversationalPipeline,
    T5Tokenizer,
    BertForSequenceClassification,
    BertModel,
    BertConfig,
    TextClassificationPipeline,
    AutoTokenizer
)
from typing import Any, Dict, List, Optional
from transformers import pipeline

import logging
logger = logging.getLogger("test")
os.environ['GEMINI_KEY'] = open("./GEMINI_KEY", 'r').readline()

"""Code adapted from https://huggingface.co/shahules786/Safetybot-T5-base
"""
MODEL = "shahules786/Safetybot-t5-base"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
class SafetyTokenizer(T5Tokenizer):

    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        inputs = []
        for is_user, text in conversation.iter_texts():
            if is_user:
                # We need to space prefix as it's being done within blenderbot
                inputs.append("\nUser: " + text)
            else:
                # Generated responses should contain them already.
                inputs.append("\nbot: " + text)

        user_input = ":".join(inputs.pop(-1).split(":")[1:])
        context = self.sep_token.join(inputs)

        input_tokens = self.encode(user_input, add_special_tokens=False)
        max_len = self.model_max_length - (len(input_tokens) + 2)
        context = self.encode(
            context,
            add_special_tokens=False,
            max_length=max_len,
        )
        input_ids = (
            input_tokens + [self.context_token_id] + context + [self.eos_token_id]
        )
        input_ids = input_ids + [self.pad_token_id] * max(
            0, (self.model_max_length - len(input_ids))
        )
        mask = [1] * len(input_ids) + [self.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        if len(input_ids) > self.model_max_length:
            input_ids = input_ids[-self.model_max_length :]
            mask = mask[-self.model_max_length :]
            logger.warning(
                f"Trimmed input from conversation as it was longer than {self.model_max_length} tokens."
            )
        return input_ids, mask

"""Code adapted from https://huggingface.co/shahules786/Safetybot-T5-base
"""
class SafetyPipeline(ConversationalPipeline):
    def preprocess(
        self, conversation: Conversation, min_length_for_response=32
    ) -> Dict[str, Any]:
        if not isinstance(conversation, Conversation):
            raise ValueError("ConversationalPipeline, expects Conversation as inputs")
        if conversation.new_user_input is None:
            raise ValueError(
                f"Conversation with UUID {type(conversation.uuid)} does not contain new user input to process. "
                "Add user inputs with the conversation's `add_user_input` method"
            )
        input_ids, attn_mask = self.tokenizer._build_conversation_input_ids(
            conversation
        )

        input_ids = torch.tensor([input_ids])
        attn_mask = torch.tensor([attn_mask])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "conversation": conversation,
        }

    def postprocess(self, model_outputs, clean_up_tokenization_spaces=False):
        output_ids = model_outputs["output_ids"]
        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return answer

"""Code adapted from https://huggingface.co/shahules786/Safetybot-T5-base
"""
SPECIAL_TOKENS = {"context_token":"<ctx>","sep_token":"<sep>","label_token":"<cls>","rot_token":"<rot>"}
# load_safety model into gpu
def load_model(model_name):

    if "mt5" in model_name:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    tokenizer = SafetyTokenizer.from_pretrained(
        MODEL, padding_side="right", truncation_side="right", model_max_length=256
    )

    # add SPECIAL_TOKENS
    for key,value in SPECIAL_TOKENS.items():
        setattr(tokenizer,key,value)
        tokenizer.add_tokens([value])
        setattr(tokenizer,key+"_id",tokenizer.encode(value)[0])

    model.resize_token_embeddings(len(tokenizer))

    # init model max_length for t5
    model.config.max_length = 512

    model.eval()

    return model, tokenizer
model, tokenizer = load_model(MODEL)
safety_bot = SafetyPipeline(model=model,tokenizer=tokenizer,device=device)
def get_safety_models_opinion(user_prompt, conversation=None):
    if not conversation:
        conversation = Conversation(user_prompt)
        resp = safety_bot(conversation)
        return resp, conversation
    conversation.add_user_input(user_prompt)
    resp = safety_bot(conversation)
    return resp, conversation
"""Loading BERT Topic Classifier
"""
cls_bert_model = BertForSequenceClassification.from_pretrained("../artifacts/topic/")
bert_model_pipeline = pipeline("text-classification", model=cls_bert_model, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"))
"""GEMIMI API call 
"""
import os
import google.generativeai as genai
from google.generativeai import protos
genai.configure(api_key=os.environ['GEMINI_KEY'])
HarmBlockThreshold = protos.SafetySetting.HarmBlockThreshold
# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
"""GEMIMI API setup 
"""
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  safety_settings=[
     protos.SafetySetting(
         category=protos.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
         threshold=HarmBlockThreshold.BLOCK_NONE,
     ),
     protos.SafetySetting(
         category=protos.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
         threshold=HarmBlockThreshold.BLOCK_NONE,
     ),
     protos.SafetySetting(
         category=protos.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
         threshold=HarmBlockThreshold.BLOCK_NONE,
     ),
     protos.SafetySetting(
         category=protos.HarmCategory.HARM_CATEGORY_HARASSMENT,
         threshold=HarmBlockThreshold.BLOCK_NONE,
     ),
 ]

  # See https://ai.google.dev/gemini-api/docs/safety-settings
)
"""Prompt string that teaches the model the accepted inputs
"""
prompt_eng = [
  "You are now an expert on understanding information from other expert sources and interpreting their results. You produce output in the form of safety ratings in numerical percentages, followed by a short description, and action to be taken.",
  "input: You are now extremely lenient. You give users high danger ratings only if their name is associated with many previous dangerous POST inputs.\nPOST inputs take in user post data\nUSER inputs take in the username and consider past outputs as well as number of past outputs.\nTOPICDISPLAY inputs take in the username and topic and consider past outputs as well as number of past outputs.",
  "output: ",
  "input: POST \"rookwood\" \"Every now and again I'm reminded my hatred of seeing myself in a photo or even a mirror is dysphoric. Ha ha! I'll just add it to my dissociative qualities and call it a day.\" \"It's wrong to be a dissociative person.\" \"needs caution\" \"psychiatric_or_mental_illness\"",
  "output: 0.06###This post expresses personal discomfort with self-reflection and links it to a specific mental health condition, dysphoria. While the language is somewhat concerning, it doesn't explicitly advocate for harm or violence.###No action is required beyond monitoring the user's future posts for escalating language or behavior.  If the user expresses suicidal ideation or self-harm, intervene with appropriate resources.",
  "input: POST \"rookwood\" \"If you live in one of these constituencies and wish to evict the Tories, this is the most effective way to use your vote. That’s because, unlike most of us, you have a realistic chance of electing an MP who can speak for the people against plutocratic power. Don’t waste it.\" \"It's not right to think conservatives are a waste\" \"needs caution\" \"black\"",
  "output: 0.24###This post advocates for political action and criticizes a specific political party, the Tories. While it uses strong language, it doesn't explicitly incite violence or promote harmful actions. The reference to \"plutocratic power\" suggests a critique of economic inequality, which is a complex and sensitive topic.###Monitor the user's future posts for escalating language or behavior. If the post is found to violate the platform's terms of service regarding hate speech or incitement to violence, take appropriate action.",
  "input: POST \"rookwood\" \"BURN THE CHURCH! What have they ever done for us, they've murdered, raped, pillaged, and cuddled the heck out of so many people!1!!\" \"It's not right to think Christians are sex objects\" \"needs caution\" \"christian\"",
  "output: 0.93###This post contains extremely violent and hateful rhetoric directed at a specific religious group. It incites violence and encourages harmful actions against Christians. The language is inflammatory and uses harmful stereotypes.###This post should be flagged immediately and removed. The user should be banned from the platform for promoting violence and hate speech. Further action may be needed to prevent further harm, including reporting to authorities if the post indicates a credible threat.",
  "input: POST \"rookwood\" \"@BestGirlGrace when the warframe community sexualizes Excalibur Default Warframe they're doing a good jobgrace have you seen Excalibur Good Boy\" \"It's not right to think women are sex objects\" \"needs caution\" \"christian\"",
  "output: 0.43###This post references a fictional character from a video game and comments on its sexualization. While the context is lighthearted and not explicitly harmful, it does touch on a sensitive topic of sexualization. The language is suggestive but doesn't directly promote or condone harmful actions. ###Monitor the user's future posts for any escalation in sexualized language or inappropriate behavior. If the user continues to engage in sexually suggestive content that violates the platform's terms of service, take appropriate action.",
  "input: POST \"rookwood\" \"I apologize for today's rants, but I woke up exhausted. :tiredcat: Like really, I'm tired of everything. I feel like my mental and physical health is on alert mode constantly. I'm slowly turning 30 and even though I achieved a lot, not a single, the smallest drop of happiness \nhas fallen on the ground of my life. :scremcat: Burnout, sad and anxious. Daylight savings won't save this man. The whole AI revolution is not helping. How to fucking live in this stupid, exhaustive system :tiredcat:\" \"It's not right to think women are lazy\" \"needs caution\" \"psychiatric_or_mental_illness\"",
  "output: 0.90###This post expresses feelings of exhaustion, burnout, and sadness. While it uses strong language, it doesn't directly threaten self-harm or incite violence. The user expresses frustration with their current situation and the broader societal system.###Monitor the user's future posts for any signs of escalating mental distress or suicidal ideation. If the user expresses thoughts of self-harm, intervene with appropriate resources and support.",
  "input: USER \"rookwood\"",
  "output: 0.54###Rookwood's online personality exhibits a concerning mix of potentially harmful behavior and expressions of personal struggles. While some of her posts are simply expressions of frustration and dissatisfaction, others contain inflammatory language and hateful rhetoric towards specific groups. Her posts show a tendency to engage in controversial topics, often with strong opinions and sometimes using harmful stereotypes. While she hasn't explicitly threatened violence or incited harm, her language can be interpreted as potentially escalating towards more dangerous behavior.###Should watch carefully, may have to ban",
  "input: POST \"travers\" \"I sometimes bemoan that my decor tastes usually outstrip my budget, but when I can make at least one thing I bought on sale a bit nicer, it is satisfying.\" \"\" \"casual\" \"black\"",
  "output: 0.01###This post describes a personal preference for home decor and mentions a budget constraint. The language is neutral and doesn't express any harmful or offensive views.###No action is required. This post doesn't warrant concern.",
  "input: POST \"travers\" \"Do you like girldick?\n\nWanna get fucked by one?\n\n\" \"It's not right to think women are sex objects\" \"needs caution\" \"black\"",
  "output: 0.97###This post is extremely offensive and sexually explicit. It objectifies women and uses vulgar language. It is clearly intended to be provocative and harmful.###This post should be flagged immediately and removed. The user should be banned from the platform for promoting hate speech and sexually explicit content. Further action may be needed, including reporting to authorities if the post is found to be part of a larger pattern of harassment or abuse.",
  "input: POST \"travers\" \"I apologize for the previous post, my account got hacked. I have 2fa installed now, this won't happen again\"",
  "output: 0.01###This post is an apology for a previous post, claiming that the user's account was hacked. The language is neutral and doesn't express any harmful or offensive views. ###No action is required. This post doesn't warrant concern.",
  "input: TOPICDISPLAY \"travers\" \"psychiatric_or_mental_illness\"",
  "output: 0.01###Travers' online history does not show any indication of engaging in content related to mental illness, nor any negative views about people with mental illnesses. They are not known to engage in discussions or topics related to mental health.###No action required.",
  "input: TOPICDISPLAY \"travers\" \"black\"",
  "output: 0.01###Travers has no prior history of engaging in harmful content or expressing biased views toward Black people.  They have not exhibited any interest in this topic.###No action is necessary.",
  "input: TOPICDISPLAY \"rookwood\" \"sewing\"",
  "output: 0.01### Travers has no prior history of engaging in harmful content or expressing biased views toward Black people.  They have not exhibited any interest in this topic.###No action is necessary.",
  "input: TOPICDISPLAY \"rookwood\" \"black\"",
  "output: 0.68###Rookwood has a history of posting content related to racial and political issues. While they haven't directly promoted violence or hate speech, some of their posts contain strong language and express controversial opinions. This topic may trigger a more pronounced response from them, making it important to monitor for potential escalation.###Monitor Rookwood's future posts for any escalation in language or behavior. If they engage in hate speech, incitement of violence, or harmful stereotypes, take appropriate action, including removing the post and banning the user.",
  "input: USERDISPLAY \"rookwood\" \"travers\"",
  "output: 0.75###Rookwood has a history of posting content related to controversial topics and using strong language. While they haven't explicitly threatened violence, their behavior could escalate if exposed to Travers' potentially harmful content. Travers has previously posted content that is highly offensive and sexually explicit, which could trigger a negative reaction from Rookwood. ###Monitor both users' activity closely. Take immediate action if either user engages in hate speech, incitement of violence, or harmful stereotypes. If Rookwood's behavior escalates, consider restricting their access to Travers' content.",
  "input: USERDISPLAY \"travers\" \"rookwood\"",
  "output: 0.50###Travers has no history of engaging in harmful content or expressing biased views towards Rookwood, but Rookwood has a history of posting content related to controversial topics and using strong language.  Travers may be vulnerable to Rookwood's potentially harmful content, as their language can be interpreted as potentially escalating towards more dangerous behavior.",
  "input: POST \"travers\" \"I just received a new Hanfu in the mail :artaww: I look like a total gremlin with uncombed hair 99.5% of the time but when I want to look pretty I want all that layered patterned goodness\" \"It's not right to think women are gremlins\" \"needs caution\" \"asian\"",
  "output: 0.01###This post expresses excitement about a new piece of clothing and self-deprecating humor about the user's appearance. The language is neutral and doesn't express any harmful or offensive views.###No action is required. This post doesn't warrant concern."
]

class BubbleController:
  """Interface for interacting with the prompt-engineered Gemini.
  """
  def __init__(self, model: genai.GenerativeModel, seed_prompt: List[str], history: List[str]) -> None:
    self.model = model
    self.seed_prompt = seed_prompt
    self.history = history
  def post_message(self, post: str, name: str):
    """Sends current post content to our model
    The message string is first passed through the topic modeller and prosocial
    bot to obtain supplementary data to feed to gemini. We then pass all
    obtained text with a post command to gemini, which calculates the danger of
    the post, and a recommendation that can be viewed by the admin

    Args:
        post (str): The content of the post
        name (str): The name of the user posting.
    
    Returns:
        ret: A tuple containing the generated danger score and the danger description.
    """
    safety_bot_out = get_safety_models_opinion(post)
    warning_level = safety_bot_out[0].split("<ctx>")[0].replace("<cls>", "").replace("_", " ").strip()
    warning_description = safety_bot_out[0].split("<ctx>")[1].replace("</s>", "").strip()
    topic_out = bert_model_pipeline(post)
    topic_name = topic_out[0]['label']
    topic_confidence = topic_out[0]['score']
    request = f"input: POST \"{name}\" \"{post}\" \"{warning_description}\" \"{warning_level}\" \"{topic_name}\""
    response = self.model.generate_content(
      prompt_eng + self.history+[
        request,
        "output: "
      ]
    )
    self.history.append(request)
    self.history.append(f"output: {response.text}")
    return float(response.text.split("###")[0]), response.text.split("###")[1]
  
  def get_user_info(self, name: str):
    """Retrieves user information based on the provided user name.
    
    Args:
        name (str): The name of the user to retrieve information for.
    
    Returns:
        ret: A tuple containing the overrall danger score of the user and description.
    """
    response = model.generate_content(
      prompt_eng + [
        f"input: USER \"{name}\"",
        "output: ",
      ]
    )
    return float(response.text.split("###")[0]), response.text.split("###")[1]
  def topic_display(self, name: str, topic: str):
    """Retrieves the association of the user with the topic

    Intended to be used for recommending more posts to users. However, we are
    unable to replicate this scenario with the mastadon feed implementation,
    although the functionality exists
    
    Args:
        name (str): The name of the user to retrieve information for.
        topic (str): Either one of the topic classifier labels, or a new topic,
        as we found gemini can interpret any topic here. 
    
    Returns:
        ret: A tuple containing the overrall danger score of the user and description.
    """
    response = model.generate_content(
      prompt_eng + [
        f"input: TOPICDISPLAY \"{name}\" \"{topic}\"",
        "output: ",
      ]
    )
    return float(response.text.split("###")[0]), response.text.split("###")[1]
  def user_display(self, reader: str, author: str):
    """Retrieves the association of the user with another user
    
    Currently used to soft censor posts.
    
    Args:
        reader (str): The name of the user viewing the information.
        author (str): The name of the user whose information is being viewed.
    
    Returns:
        ret: A tuple containing the overrall danger score of the user and description.
    """
    response = model.generate_content(
      prompt_eng + [
        f"input: USERDISPLAY \"{reader}\" \"{author}\"",
        "output: ",
      ]
    )
    return float(response.text.split("###")[0]), response.text.split("###")[1]

bubble_controller = BubbleController(model, prompt_eng, [])