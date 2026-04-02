import streamlit as st
import numpy as np
import re
from preprocessing import input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens
from training_model import decoder_inputs, decoder_lstm, decoder_dense, num_decoder_tokens, latent_dim

from keras.layers import Input
from keras.models import Model, load_model

# --------------------------------------------------
# Load model and build encoder/decoder
# --------------------------------------------------
@st.cache_resource
def load_models():
    training_model = load_model('training_model.h5')
    
    encoder_inputs = training_model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

encoder_model, decoder_model = load_models()

# --------------------------------------------------
# Encode user input into one-hot matrix
# --------------------------------------------------
def encode_input(sentence):
    tokenized = re.findall(r"[\w']+|[^\s\w]", sentence)
    input_matrix = np.zeros((1, len(tokenized), num_encoder_tokens))
    for t, token in enumerate(tokenized):
        if token in input_features_dict:
            input_matrix[0, t, input_features_dict[token]] = 1.
    return input_matrix

# --------------------------------------------------
# Decode sequence using the trained model
# --------------------------------------------------
def decode_sequence(test_input):
    states_value = encoder_model.predict(test_input, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    decoded_sentence = ''
    stop_condition = False

    while not stop_condition:
        output_tokens, hidden_state, cell_state = decoder_model.predict(
            [target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]

        if sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_token

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [hidden_state, cell_state]

    return decoded_sentence.strip()

# --------------------------------------------------
# Rule-based fallback for common inputs
# --------------------------------------------------
def rule_based_response(user_input):
    text = user_input.lower().strip()

    greetings = ["hi", "hello", "hey", "howdy", "greetings"]
    if text in greetings:
        return "Hello! How are you doing today?"

    farewells = ["bye", "goodbye", "see you", "farewell", "see ya"]
    if text in farewells:
        return "Goodbye! Have a great day!"

    thanks = ["thanks", "thank you", "thx", "cheers"]
    if text in thanks:
        return "You're welcome!"

    if text in ["how are you", "how are you?", "how are you doing"]:
        return "I'm doing well, thank you for asking! How about you?"

    if text in ["what is your name", "what is your name?", "who are you", "who are you?"]:
        return "I'm a chatbot built with an LSTM neural network!"

    return None

# --------------------------------------------------
# Combined response: rule-based first, then model
# --------------------------------------------------
def get_response(user_input):
    # Try rule-based first
    rule_response = rule_based_response(user_input)
    if rule_response:
        return rule_response, "rule-based"

    # Fall back to neural network
    encoded = encode_input(user_input)
    response = decode_sequence(encoded)
    return response, "neural network"

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("LSTM Chatbot")
st.caption("A generative chatbot using encoder-decoder architecture with LSTM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Type a message..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response
    response, method = get_response(user_input)

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.caption(f"Response method: {method}")
    st.session_state.messages.append({"role": "assistant", "content": response})