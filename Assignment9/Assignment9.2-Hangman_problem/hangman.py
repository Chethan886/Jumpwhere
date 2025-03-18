import random

# Function to read words from file and store them in a list
def load_words(filename):
    with open(filename, "r") as file:
        words = [line.strip().upper() for line in file]  # Convert words to uppercase
    return words

# Function to choose a random word from the list
def choose_word(words):
    return random.choice(words)

# Function to play Hangman
def play_hangman():
    words = load_words(r"c:/Users/asus/Desktop/assignment0/Jumpwhere/Assignment9.2-Hangman_problem/words.txt")
    
    while True:
        secret_word = choose_word(words)  # Select a random word
        guessed_letters = set()  # Store guessed letters
        incorrect_guesses = 0
        max_attempts = 6

        print("\nWelcome to Hangman!")
        display_word = ["_" for _ in secret_word]  # Create the masked word
        print(" ".join(display_word))

        while incorrect_guesses < max_attempts:
            guess = input("\nGuess your letter: ").upper()

            if guess in guessed_letters:
                print("You already guessed that letter. Try again!")
                continue

            guessed_letters.add(guess)

            if guess in secret_word:
                for i, letter in enumerate(secret_word):
                    if letter == guess:
                        display_word[i] = guess
                print(" ".join(display_word))
            else:
                incorrect_guesses += 1
                print(f"Incorrect! You have {max_attempts - incorrect_guesses} chances left.")

            if "_" not in display_word:
                print("\n🎉 Congratulations! You guessed the word correctly.")
                break
        else:
            print(f"\n💀 Game Over! The word was: {secret_word}")

        # Ask if the user wants to play again
        play_again = input("\nDo you want to play again? (yes/no): ").lower()
        if play_again != "yes":
            print("Thanks for playing Hangman! Goodbye!")
            break

# Run the game
play_hangman()
