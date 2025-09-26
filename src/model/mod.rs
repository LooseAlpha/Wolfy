mod net_printer;
mod async_tasks;

use std::{fs::File, io::{BufReader, BufWriter, Error, Write}, str::FromStr, thread};
use std::fmt;
use rand::{distr::StandardUniform, prelude::*};
use crate::model::net_printer::*;
#[allow(unused_imports)]
use crate::echo;

//could async be used to do multiple modification tasks/training/testing 
//at the same time? Im imagining a complex tree structure
//with regions of improvement all happening in parallel. 
//sounds like a complex waker system.
//maybe break down tasks into using async
//services queue so they can freely overlap.

//Im thinking that my data storage needs completely rethought.
//perhaps the net holds rings that simply have slices into 
//a weight repository. usize indexes would still be used for
//the slices instead of pointers, but those slices would be
//the basic units we offer the machine to do work.
//Do what? forward prop
//here is the input to use
//here is the operation to use
//here are the weights to use
//here are the biases to use
//here is the Relu to use
//
//So, each ring is a job...
//

type Weights<W> = Vec<W>;
type Biases<B> = Vec<B>;
type Funnel<N> = Vec<N>;
type Layer<L> = Vec<L>;
type Activations<A> = Vec<A>;

// The Model holds all the persistent state in mirrored vector trees.
// Layers -> Rings -> Neurons
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Model {
    ring_net: Funnel<Layer<Ring>>,
    weight_net: Funnel<Layer<Weights<f32>>>,
    bias_net: Funnel<Layer<Biases<f32>>>,
    state: ModelState,
}

pub struct Ticket(u64, Destination);
pub enum Destination {
    Answer,//this
    Evaluate,//this
    Train,//on this
}
// The position of each Ring in the repository is that rings identity
// It's Weights and Biases can be found under the same index in those respective repositories
//
// rework: Each ring is an async task. Stored in a 1 dimensionalized queue.
// One rings output is a waker to one or more rings. One ring
// may wait on multiple inputs to wake. What about the backprop? Is
// it a whole new task ring system or a supervisor mode? We need
// mut access to the same data. 
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Ring { 
    children: Vec<Child>, 
    //the number of output "pixels"
    neuron_count: usize, 
    //activation_function: enum,
}

//the answer digits are the parent neurons
//all above are children
//each layer made up of rings. subdivision of layers.
//all neurons in a ring have same children rings. 
//backprop only happens on ringsets of mutual children(?)
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Debug, Clone)]
pub struct Child(pub usize, pub usize);

#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Debug, Clone)]
enum ModelState {
    Malformed,
    Functional,
}

impl Model {
    pub fn startup() -> Model {
        let file = String::from_str("workbench").unwrap();
        if let Ok(model) = Model::load(&file) { model } 
        else { Model::new(4, 4) }
    }
    pub fn new(depth: usize, width: usize) -> Model {
        //create backbone
        let (
            mut ring_net, 
            mut weight_net, 
            mut bias_net
            ) = Model::new_backbone();
        
        //create initial layers
        Model::new_initial_layers(&mut ring_net, &mut weight_net, &mut bias_net);
        
        //breaking things down into step functions hardly saves me space.
        //I expected visual simplicity:
        //do_this();
        //do_that();
        
        //create starting ring
        let mut children = Vec::new();
        children.push(Child(0,0));
        ring_net[0].push(
            Ring {
                children,
                neuron_count: 28*28,
            }
        );
        
        //create all layers
        for l in 0..depth {
            //Vecs dont touch the allocator until the first push
            //only ring_layer + children hit the allocator 0..depth times
            let mut ring_layer = Vec::new();
            let weight_layer = Vec::new();
            let bias_layer = Vec::new();
            
            let mut children = Vec::new();
            children.push(Child(l,0));
            ring_layer.push(Ring {
                    children,
                    neuron_count: width,
                }
            );
            ring_net.push(ring_layer);
            weight_net.push(weight_layer);
            bias_net.push(bias_layer);
        }
        
        //for each layer dispatch threads
        thread::scope(|s|{
            // [i] always takes full mutability of the entire buffer.
            // when the type system controls the access, we can safely send mutability.
            // when we generate the index, rust cant determine we did the math correctly.
            // even if it was generated with an iterator (0..range).
            // we step out of the chain_of_trust to wire up the index correctly[i].
            // different domains of control, type vs computation.
            
            //  create weights
            for (mut_layer, read_layer) in weight_net.iter_mut().zip(&ring_net).skip(1) {
                s.spawn(|| {
                    let weight_count = Model::source_activations_count(
                        &ring_net, 
                        &read_layer[0]
                        ) * read_layer[0].neuron_count;
                    let mut matrix = Vec::new();
                    for _ in 0..weight_count {
                        matrix.push((rand::rng().sample::<f32, _>(StandardUniform) - 0.5 ) * 2.0);
                    }
                    mut_layer.push(matrix);
                });
            }
            //  create biases
            for (mut_layer, read_layer) in bias_net.iter_mut().zip(&ring_net).skip(1) {
                s.spawn(|| {
                    let bias_count = read_layer[0].neuron_count;
                    let mut matrix = Vec::new();
                    for _ in 0..bias_count {
                        matrix.push((rand::rng().sample::<f32, _>(StandardUniform) - 0.5 ) * 2.0);
                    }
                    mut_layer.push(matrix);
                });
            }
            //is the allocator atomic?
            //perhaps the available memory ledger is behind a mutex. 
            
            //Internal Locking: The allocator's internal data structures (e.g., free lists, block metadata) are 
            //protected by synchronization primitives like mutexes or spinlocks. This ensures that only one thread 
            //can modify these structures at a time, preventing corruption.
            
            // spinlocks would make more sense.
        });
        Model {
            ring_net,
            weight_net,
            bias_net,
            state: ModelState::Malformed,
        }
        
        //.. huh... it works. 
        // created 1000x1000 in a second
        // thats each layer with 1000 neurons with 1000 weights each
        // 1 million weights per layer with 1000 layers.
        // 1 billion f32 values in about a second. 
        // 1000 threads were spawned. 
        // didnt even preallocate the vec sizes needed.
        // each followed their growth curve, hitting alloc every resize.
        // Well past the size of pages in mem the alloc receives.
        // single threaded 1000x1000 took like 10sec. still impressive.
        // single threaded save_to_file takes like 20.
        // lets not bother testing the training algo at this scale. 
        // also, this probably isnt more performant at the 
        // scales the network will normally operate in. I couldnt see
        // how I would care about such a small degredation. Id see 
        // an issue with deep and narrow nets e.g.[1000]x[10]. Too 
        // little work for each thread. Still too high a tolerance.
    }
    
    pub fn test(&self, data: &Vec<Vec<f32>>, labels: &Vec<u8>) {
        let mut correct = 0usize;
        let mut incorrect = 0usize;
        for index in 0..data.len() {
            let image = &data[index];
            let answer = labels[index] as usize;
            let results = self.projection(&image);
            if results == answer {
                correct += 1;
            } else {
                incorrect += 1;
            }
        }
        println!("correct:   {correct}");
        println!("incorrect: {incorrect}");
    }
    pub fn save(&self, file: &String) -> Result<(), Error> {
        let mut path = file.clone();
        path.push_str(".json");
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }
    pub fn load(file: &String) -> Result<Model, Error> {
        let mut path = file.clone();
        path.push_str(".json");
        let file = File::open(&path)?;
        let rdr = BufReader::new(file);
        let model: Model = serde_json::from_reader(rdr)?;
        Ok(model)
    }
    pub fn exit(&self) -> ! {
        println!(">shutting down");
        let _ = self.save(&"workbench".to_owned());
        //shut down sequence
        std::process::exit(0)
    }
    pub fn source_activations_count(ring_net: &Vec<Vec<Ring>>, ring: &Ring) -> usize {
        let children = &ring.children;
        let mut activations_count = 0;
        for child in children {
            activations_count += ring_net[child.0][child.1].neuron_count;
        }
        activations_count 
    }
    fn new_backbone<T, U, V>() -> (Vec<T>, Vec<U>, Vec<V>) {
        (Vec::new(), Vec::new(), Vec::new())
    }
    fn new_initial_layers(
        ring_net: &mut Vec<Vec<Ring>>, 
        weight_net: &mut Vec<Vec<Vec<f32>>>,
        bias_net: &mut Vec<Vec<Vec<f32>>>
        ) {
        ring_net.push(Vec::new());
        weight_net.push(Vec::new());
        bias_net.push(Vec::new());
    }
    
    //these are pretty good 1 day goals. 
    //maybe clean up the command UX too.
    //while writing them, write them to use with commands
    //but think of how an async system would need to interface.
    pub fn add_ring(&mut self, layer: usize, neuron_count: usize, children: Vec<Child>) {
        assert!(layer <= self.ring_net.len());
        if layer == self.ring_net.len() {
            self.ring_net.push(Vec::new());
            self.weight_net.push(Vec::new());
            self.bias_net.push(Vec::new());
        }
        assert_eq!(&self.ring_net[layer].len(), &self.weight_net[layer].len());
        assert_eq!(&self.ring_net[layer].len(), &self.bias_net[layer].len());
        
        let ring = Ring {
            children,
            neuron_count,
        };
        let weight_count = Model::source_activations_count(&self.ring_net, &ring);
        let total_weight_count = weight_count * neuron_count;
        let mut w_matrix = Vec::new();
        let mut b_matrix = Vec::new();
        for _ in 0..total_weight_count {
            w_matrix.push((rand::rng().sample::<f32, _>(StandardUniform) - 0.5 ) * 2.0);
        }
        for _ in 0..neuron_count {
            b_matrix.push((rand::rng().sample::<f32, _>(StandardUniform) - 0.5 ) * 2.0);
        }
        
        self.ring_net[layer].push(ring);
        self.weight_net[layer].push(w_matrix);
        self.bias_net[layer].push(b_matrix);
        
        //how do the properties of rings live and interact?
        //Relu, SoftMax, many_hot, targetting...
    }
    fn add_children(&mut self, ring: (usize, usize), children: &Vec<Child>) {
        //changes weights
        
        // I should have had the accessor to this net structure be an api.
        // Something simple that hides more complexity. 
        // What is the only thing I care about at each step?
        // That should be the only thing I'm dealing with. 
        
        let (l, r) = ring;
        let neuron_count = self.ring_net[l][r].neuron_count;

        let mut old_weights = Vec::new();
        let old_input_count = Model::source_activations_count(&self.ring_net, &self.ring_net[l][r]);
        let mut current_weights = self.weight_net[l][r].chunks_exact(old_input_count);
        for _ in 0..neuron_count {
            let matrix = current_weights.next().unwrap().to_owned();
            old_weights.push(matrix);
        }
        
        for child in children {
            //add child to ring
            self.ring_net[l][r].children.push(child.clone());
        }
        let new_input_count = Model::source_activations_count(&self.ring_net, &self.ring_net[l][r]);
        let additional = new_input_count.checked_sub(old_input_count);
        let additional = additional.unwrap();
        
        //I like the idea of creating my own iterators like this.
        //not sure if it was useful here. 
        let initializer = std::iter::from_fn(
            ||{
                let new_random_weight = (rand::rng().sample::<f32, _>(StandardUniform) - 0.5 ) * 2.0;
                Some(new_random_weight)
            }).take(additional);
            
        let mut new_weights = Vec::new();
        for _ in 0..neuron_count {
            let matrix = initializer.clone().collect::<Vec<f32>>();
            new_weights.push(matrix);
        }
        let mut collected_weights = Vec::new();
        for i in 0..neuron_count {
            collected_weights.push(old_weights[i].clone());
            collected_weights.push(new_weights[i].clone());
        }
        self.weight_net[l][r] = collected_weights.into_iter().flatten().collect::<Vec<f32>>();
        
    }
    
    //removing mid tree rings is the ultimate goal.
    //
    //What is each possible arrangement and their solutions?
    // 1->x->1; 2->x->1; 1->x->2; 2->x->2
    fn remove_ring(){
        todo!()
    }
    
    //our neurons are positional and externally determined. 
    //What does modifying the count even mean?
    fn modify_neuron_count(){
        todo!()
    }
    
    
    
    pub fn cap(&mut self) {
        
        //model.create_new_ring(where, from)
        
        //create single ring on new last layer of net.
        let last_layer = self.ring_net.len() - 1;
        let cap_layer = self.ring_net.len();
        let ring_count = self.ring_net[last_layer].len();
        let mut children = Vec::new();
        for i in 0..ring_count {
            children.push(Child(last_layer, i));
        }
        let neuron_count = 10;
        let ring = Ring {
            children,
            neuron_count,
        };
        self.ring_net.push(Vec::new()); //new last layer
        self.ring_net[cap_layer].push(ring); //new solo ring
        
        //fill the weights
        self.weight_net.push(Vec::new());
        let weight_count = Model::source_activations_count(&self.ring_net, &self.ring_net[cap_layer][0]);
        let mut matrix = Vec::new();
        for _ in 0..neuron_count { // for 10 neurons in our cap ring
            for _ in 0..weight_count { //multiply weights by neuron count
                matrix.push((rand::rng().sample::<f32, _>(StandardUniform) - 0.5 ) * 2.0);
            }
        }
        self.weight_net[cap_layer].push(matrix);
        //fill biases
        let matrix = vec![0f32;10];
        self.bias_net.push(Vec::new());
        self.bias_net[cap_layer].push(matrix);
        self.state = ModelState::Functional;
        println!(">Cap layer created");
        self.display_model();
    }
    pub fn display_model(&self) {
        let mut net_printer = NetPrinter::new();
        net_printer.display_net(&self);
    }
    //allocating new activation net for each image
    //perhaps take in a reference to the dataset
    //and a range to work through. maybe in test.
    fn projection(&self, image: &Vec<f32>) -> usize { //refactor because redesign
        //single wave through funnel
        
        //each ring:
        //workpiece.input()
        //      which input? defined by children pointers
        //workpiece.biases()
        //workpiece.function()
        //      which function? saved in ring as well.
        
        let ring_net = &self.ring_net;
        // make sure the model is well formed
        match self.state {
            ModelState::Malformed => return usize::MAX,
            _ => (),
        }
        
        
        assert_eq!(image.len(), 28*28);
        let mut activation_net: Funnel<Layer<Activations<f32>>> = Vec::new();
        let mut initial_layer = Vec::new();
        initial_layer.push(image.clone());
        activation_net.push(initial_layer);
        let mut answer = (usize::MAX, &-1.0f32);
        answer.0
        
        
        
        
    }
    fn reflection(){
        //wave through funnel that is reflected back
        //one_hot encoding, with our math for error,
        //is our mirror. backprop
    }
}


impl fmt::Display for Ring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, 
            "Ring {{ children: {:?}, weights: ..., biases: ..., neuron_count: {} }}",
            self.children,
            self.neuron_count,
        )
    }
}

//stop naming the collections as a plural of their item.
//