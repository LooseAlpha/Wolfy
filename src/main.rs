mod model;

use crate::model::*;
use std::{fmt::Debug, io::{self}};
use mnist_reader::*;
use tokio::*;

#[tokio::main]
fn main() {
    let mut model = Model::startup();
    let _intermediate: Vec<Vec<Vec<f32>>> = Vec::new();
    
    // read MNIST data
    let mut mnist = MnistReader::new("mnist-data");
    // download MNIST data
    mnist.load().unwrap();
    // print the size of the data
    println!("Train data size: {}", mnist.train_data.len());
    println!("Test data size: {}", mnist.test_data.len());
    println!("Train labels size: {}", mnist.train_labels.len());
    println!("Test labels size: {}", mnist.test_labels.len());

    let train_data = mnist.train_data;
    let train_labels = mnist.train_labels;
    let test_data = mnist.test_data;
    let test_labels = mnist.test_labels;
    
    let mut buf = String::new();
    loop {
        println!("Command: ");
        io::stdin().read_line(&mut buf).expect(">failed to read input");
        buf = buf.trim_end().to_string();
        match buf.as_str() {
            "exit" => model.exit(),
            "help" => help(),
            "new" => new_model_system(&mut model),
            "save" => save_model_system(&model),
            "load" => load_model_system(&mut model),
            "display" => data_view_system(&train_data, &train_labels),
            "show" => model.display_model(),
            "shape" => shape_system(&mut model),
            "test" => test_system(&model, &test_data, &test_labels),
            "train" => train_system(&mut model, &train_data, &train_labels),
            "kwatz!" => (),
            other => println!(">{other} not a command"),
        }
        buf.clear();
    }
}
//this isn't as usable as I'd like
//and nothing stops me from missing 
//implementating one of them. 
static COMMANDS: [&str; 21] = [
    " help",
    " exit",
    " show",
    " new",
    "   cancel",
    "   help",
    " save",
    "   cancel",
    "   help",
    " load",
    "   cancel",
    "   help",
    " display",
    "   cancel",
    "   help",
    " shape",
    "   cap",
    "   add ring",
    "     end",
    "   cancel",
    "   help",
];
fn save_model_system(model: &Model) {
    println!("(\"cancel\" to abort) \nfilename:");
    let mut newbuf = String::new();
    io::stdin().read_line(&mut newbuf).expect(">failed to read input");
    newbuf = newbuf.trim_end().to_string();
    match newbuf.as_str() {
        "cancel" => { println!(">canceling save operation"); return },
        _ => {
            if let Ok(result) = model.save(&newbuf) {
                result
            } else {
                println!(">failed to save model");
                return
            }
            println!(">model saved to file {newbuf}.json");
        }
    }
}
fn load_model_system(model: &mut Model)  {
    println!("(\"cancel\" to abort) \nfilename:");
    let mut newbuf = String::new();
    io::stdin().read_line(&mut newbuf).expect(">failed to read input");
    newbuf = newbuf.trim_end().to_string();
    match newbuf.as_str() {
        "cancel" => { println!(">canceling load operation"); return },
        _ => {
            if let Ok(result) = Model::load(&newbuf) {
                *model = result;
            } else {
                println!(">failed to load model");
                return
            }
            println!(">model loaded from file {newbuf}.json");
        }
    }
}
fn help() {
    println!(">Command list:");
    for str in COMMANDS {
        println!("{str}");
    }
}
fn new_model_system(model: &mut Model) {
    
    println!("New Model System");
    let depth;
    loop {
        println!("enter net depth");
        let input = user_input();
        match input.as_str() {
            "cancel" => { println!(">canceling new_model operation"); return },
            "exit" => { println!(">canceling new_model operation"); return },
            other => {
                if let Ok(d) = other.parse::<usize>() {
                    depth = d;
                    break
                } else {
                    println!(">invalid number");
                    continue
                }
            }
        }
    }
    let width;
    loop {
        println!("enter net width");
        let input = user_input();
        match input.as_str() {
            "cancel" => { println!(">canceling new_model operation"); return },
            "exit" => { println!(">canceling new_model operation"); return },
            other => {
                if let Ok(d) = other.parse::<usize>() {
                    width = d;
                    break
                } else {
                    println!(">invalid number");
                    continue
                }
            }
        }
    }
    *model = Model::new(depth, width);
    model.display_model();
    println!(">new model created");
}
fn data_view_system(data: &Vec<Vec<f32>>, labels: &Vec<u8>) {
    loop {
        println!("Data View System");
        println!("enter index");
        let input = user_input();
        match input.as_str() {
            "cancel" => { println!(">canceling display operation"); return },
            "exit" => { println!(">canceling display operation"); return },
            num => {
                let index = num.parse::<usize>();
                if index.is_err() { println!(">malformed index"); continue }
                let index = index.unwrap();
                if index >= data.len() { 
                    println!(">index {} exceeds maximum of {}", index, data.len() - 1);
                    continue 
                    }
                let data = &data[index];
                for n in (0..=26).step_by(2) {
                    let mut line = String::new();
                    for i in 0..28 {
                        if data[i + (n*28)] > 0.5 {
                            if data[i + (n*28+28)] > 0.5 {
                                line.push('█');
                            } else {
                                line.push('▀');
                            }
                        } else {
                            if data[i + (n*28+28)] > 0.5 {
                                line.push('▄');
                            } else {
                                line.push(' ');
                            }
                        }
                    }
                    if n == 0 {
                        let ch = (labels[index] + b'0') as char;
                        line.push(ch);
                    }
                    println!("{line}");
                }
            }
        }
    }
}

fn user_input() -> String {
    println!("Command: ");
    let mut newbuf = String::new(); 
    io::stdin().read_line(&mut newbuf).expect(">failed to read input");
    newbuf = newbuf.trim_end().to_string();
    newbuf
}

fn train_system(model: &mut Model, _data: &Vec<Vec<f32>>, _labels: &Vec<u8>) {
    loop {
        println!("Train System");
        let input = user_input();
        match input.as_str() {
            "cap" => model.cap(),
            "exit" => return,
            "cancel" => return,
            other => { println!(">{other} is not a shape command"); continue }
        }
    }
    //one_hot encoding. 
        let mut one_hot = Vec::new();
        for label in _labels {
            let mut e = [0.0;10];
            let i = *label as usize;
            if i >= 10 {panic!("out of bounds digit")}
            e[i] = 1.00;
            one_hot.push(e);
        }
        println!("{:?}", one_hot[0]);
}

fn child_system(ring_layer: usize) -> Option<Vec<Child>> {
    let mut result: Vec<Child> = Vec::new();
    let mut count= 0;
    loop {
        println!("Child double [{count}]: ");
        let input = user_input();
        let input = input.as_str();
        match input {
            "cancel" => return None,
            "exit" => return None,
            "end" => return Some(result),
            other => {
                let mut double = other.split(' ');
                let layer = double.next().unwrap();
                let ring = double.next().unwrap();
                assert_eq!(double.next(), None);
                let layer = layer.parse::<usize>();
                let ring = ring.parse::<usize>();
                if layer.is_err() { println!(">malformed layer: {layer:?}"); continue }
                if ring.is_err() { println!(">malformed ring: {ring:?}"); continue }
                let layer = layer.unwrap();
                if layer >= ring_layer { 
                    println!(">Child must be of preceeding layer");
                    continue
                }
                let ring = ring.unwrap();
                result.push(Child(layer, ring));
                count += 1;
            },
        }
    }
}

fn add_ring_system(model: &mut Model){
    loop {
        println!("Layer number: ");
        let input = user_input();
        let input = input.as_str();
        match input {
            "cancel" => return,
            "exit" => return,
            other => {
                let other = other.parse::<usize>();
                if other.is_err() { println!(">malformed index: {other:?}"); continue }
                let layer = other.unwrap();
                if layer == 0 { println!(">cannot add to input layer"); continue }
                let neurons = ask_neuron_count();
                if neurons == None { 
                    print!(">no neurons to form ring");
                    return 
                }
                let neuron_count = neurons.unwrap();
                if let Some(children) = child_system(layer) {
                    model.add_ring(layer, neuron_count, children);
                    println!(">Ring added");
                    model.display_model();
                    return
                } else {
                    println!(">no children to form ring");
                    return
                }
            },
        }
    }
}

fn ask_neuron_count() -> Option<usize> {
    loop {
        println!("Neuron count: ");
        let input = user_input();
        match input.as_str() {
            "exit" => return None,
            "cancel" => return None,
            other => { 
                if let Ok(num) = other.parse::<usize>() {
                    return Some(num)
                } else {
                    println!(">malformed usize: {other}"); 
                    continue 
                }
            }    
        }
    }
}

fn shape_system(model: &mut Model) {
    loop {
        println!("Shape System");
        let input = user_input();
        match input.as_str() {
            "cap" => model.cap(),
            "add ring" => add_ring_system(model),
            "exit" => return,
            "cancel" => return,
            other => { println!(">{other} is not a shape command"); continue }
        }
    }
}

fn test_system(model: &Model, test_data: &Vec<Vec<f32>>, test_labels: &Vec<u8>) {
    loop {
        println!("Test System");
        let input = user_input();
        match input.as_str() {
            "cancel" => return,
            "exit" => return,
            "test all" => model.test(&test_data, &test_labels),
            _ => continue,
        }
    }
}

fn echo<T: Debug>(input: &T) {
    println!("{input:?}");
}

