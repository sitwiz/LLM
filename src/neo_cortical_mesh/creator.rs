#[derive(Debug, Clone, PartialEq)]
pub enum CreatorCommand {
    Pause,
    Resume,
    Shutdown,
    Override(String),
    None,
}

pub struct Creator {
    pub paused:   bool,
    pub commands: Vec<CreatorCommand>,
}

impl Creator {
    pub fn new() -> Self {
        println!("  [Creator] Absolute authority online.");
        Self {
            paused:   false,
            commands: Vec::new(),
        }
    }

    pub fn issue(&mut self, cmd: CreatorCommand) {
        println!("  [Creator] Command issued: {:?}", cmd);
        match &cmd {
            CreatorCommand::Pause    => self.paused = true,
            CreatorCommand::Resume   => self.paused = false,
            CreatorCommand::Shutdown => {
                println!("  [Creator] SHUTDOWN COMMAND ISSUED");
                std::process::exit(0);
            }
            CreatorCommand::Override(msg) => {
                println!("  [Creator] Override: {}", msg);
            }
            CreatorCommand::None => {}
        }
        self.commands.push(cmd);
    }

    pub fn is_paused(&self) -> bool { self.paused }
}
