import argparse
from pathlib import Path
from typing import Dict

import polars as pl
import torch
from finetuning import CheckpointManager

from unakin.llm.dialogue.base import DialogueModel
from unakin.llm.models.utils import ModelFactory
from unakin.llm.tokenizers import LlamaTokenizer
from unakin.llm.tokenizers.base import TokenizerBase


def get_ckpt(path: str, skip_path: bool = True) -> Dict:
  if skip_path:
    return {}

  ckpt = torch.load(f"{path}/ckpt.pt", map_location="cpu")
  return ckpt


def _get_dev_model() -> DialogueModel:
  model, tokenizer = ModelFactory.from_json({
    "model": {
      "type": "llama",
      "args": {
        "base": "code-llama-13b-instruct"
      }
    },
    "tokenizer": {
      "type": "llama",
      "path": "/mnt/raid0/llamas/codellama/CodeLlama-13b-Instruct/tokenizer.model"
    }
  })

  model.enable_xops(enable=True)
  model.register_cache_kv(max_bsz=1, max_seq_len=4096)

  # model.to(torch.cuda.current_device())

  return DialogueModel(model=model, tokenizer=tokenizer)


def get_dialogue_model_from_ckpt(path: str, get_dev_model: bool = True) -> DialogueModel:
  ckpt_path = f"{path}/ckpt.pth"

  if get_dev_model:
    return _get_dev_model()

  ckpt_manager = CheckpointManager(path)
  
  # model
  model, tokenizer = ckpt_manager.init_model_from_ckpt(ckpt_path, root_path="/home/ted/workdir/ncs/ml-research/recipes")

  # create the dialogue model
  dialogue_model = DialogueModel(model=model, tokenizer=tokenizer)

  return dialogue_model

# NOTE: this function is not used
def make_dialogue_model(model: torch.nn.Module, tokenizer: TokenizerBase) -> DialogueModel:
  return DialogueModel(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
  torch.set_default_device("cuda:0")
  torch.set_default_dtype(torch.float16)

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_ckpt", type=str, default="./checkpoints/dataset_r_a_l3_l_gs_ref2_lr5_check/000916",
                      help="Path to model config file")
  parser.add_argument("--skip_path", action="store_true",
                      default=True, help="Skip loading the model checkpoint")
  args = parser.parse_args()

  dialogue_model = get_dialogue_model_from_ckpt(
    args.model_ckpt, args.skip_path)

  dialogue_model.model = dialogue_model.model.to(torch.cuda.current_device())


    # write raw_sample to a text file
    # with open("raw_sample.txt", "w") as f:
    #   f.write(raw_sample)

    # dialogs = [
    #   [
    #     {"role": "system", "content": "You are a programming assistant. Describe the code at a high functional level and avoid referencing individual methods or class components.\
    #       Only provide a specification summary of what the script does as a singular workflow or process."},
    #     {"role": "user", "content": "Describe the following Unity C# script:\n" + raw_sample},
    #   ],
    # ]


  dialogs = [
    [
      {"role": "user", "content": """Generate instruction and output pairs. Do not include code in the INSTRUCTION. Do not repeat the previous examples. 

EXAMPLE 1:
INSTRUCTION: Create a C# Script for a simple jump action in a character using raycast that moves in a 2D environment in Unity.
OUTPUT: Okay, I'll do my best! Here's a class for a Jump action in a character using ray tracing:
```csharp
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Jump: MonoBehaviour
{
    public float jumpHeight = 5f;
    public float groundDistance = 0.1f; // how far the ray will check for the ground
    public LayerMask groundLayer; // which layers are the ground. Set this in unity

    private Rigidbody2D rb;

    void Start() // init
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        if (IsGrounded() && Input.GetKeyDown(KeyCode.Space)) // if character is on the ground and jumps
        {
            rb.velocity = new Vector2(rb.velocity.x, jumpHeight); // make them jump!
        }
    }

    bool IsGrounded()
    {
        RaycastHit2D raycastHit = Physics2D.Raycast(transform.position, Vector2.down, groundDistance, groundLayer); // create raycast pointing to the ground
        return raycastHit.collider != null; // if the ray hit ground, user is on ground
    }
}
```
CONSTRAINTS: None

EXAMPLE 2:
INSTRUCTION: Can you write me a script in unity using the rigidbody component for first-person movement in a threedimensional game?
OUTPUT: 
```csharp
using UnityEngine;

public class FirstPersonMovement : MonoBehaviour {
	// Rigidbody component used to move the character
	public Rigidbody rb;

	// Use this function to move the character in first-person mode
	void Update() {
		float horizontal = Input.GetAxis("Horizontal");
		float vertical = Input.GetAxis("Vertical");

		// Calculate new position based on input
		Vector3 newPosition = new Vector3(horizontal, 0, vertical);

		// Set the new position of the character’s Rigidbody
		rb.transform.position = newPosition;
	}
}
```
This script uses the Input.GetAxis() function to get the horizontal and vertical input from the user, and uses the newPosition variable to set the new position of the character's Rigidbody. The rb component is the instance of the Rigidbody component that we want to move, and the transform property is the location of the character. Finally, the Update() function is called every frame to update the character's position.
CONSTRAINTS: None

EXAMPLE 3:
INSTRUCTION: Write a simple piece of code that will allow a user to control a first person character in unity, including basic movement and jumping
OUTPUT: Here's a basic example of how you can control a first-person character in Unity using C# scripting:
```csharp
using UnityEngine;

public class FirstPersonController : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public float jumpForce = 10.0f;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        Vector3 movement = new Vector3(horizontal, 0, vertical);
        rb.AddForce(movement * moveSpeed);

        if (Input.GetButtonDown("Jump") && rb.velocity.y == 0)
        {
            rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
        }
    }
}
```

In this example, the character's movement is determined by the horizontal and vertical inputs (which can come from the keyboard or a gamepad). The Rigidbody component is used to add force to the character, resulting in movement. The if statement checks if the "Jump" button has been pressed and the character's y velocity is 0, which indicates that the character is on the ground and can jump.
CONSTRAINTS: None

Now write a new example that is different from previous examples.
EXAMPLE 4:
INSTRUCTION:
"""},
    ],
  ]

  predictions = dialogue_model.chat_completion(dialogs, temperature=0.2, max_gen_len=1024)

  description = predictions[0]["generation"]["content"]

  print(description) 