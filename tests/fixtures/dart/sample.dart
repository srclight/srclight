// A sample Dart file for testing symbol extraction.

import 'dart:math';

// Top-level function.
int add(int a, int b) {
  return a + b;
}

// A service class.
class UserService {
  final String _name;

  UserService(this._name);

  String get name => _name;

  /// Fetches a user by ID.
  Future<User?> fetchUser(int id) async {
    return null;
  }

  Future<List<User>> fetchAll() async {
    return [];
  }
}

// A model class.
class User {
  final int id;
  final String email;
  final UserStatus status;

  const User({
    required this.id,
    required this.email,
    this.status = UserStatus.active,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'] as int,
      email: json['email'] as String,
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'email': email,
    'status': status.name,
  };
}

// An enum for user status.
enum UserStatus {
  active,
  inactive,
  pending,
}

// A mixin for logging.
mixin Logger {
  void log(String message) {
    print('[LOG] $message');
  }
}

// A mixin for caching.
mixin CacheMixin {
  final Map<String, dynamic> _cache = {};

  dynamic get(String key) => _cache[key];

  void set(String key, dynamic value) {
    _cache[key] = value;
  }

  void clear() => _cache.clear();
}

// A class that uses a mixin.
class DataManager with Logger, CacheMixin {
  Future<void> load() async {
    log('Loading data...');
  }
}

// Extension on String.
extension StringExtensions on String {
  String get capitalized {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1)}';
  }

  bool get isBlank => trim().isEmpty;
}

// Extension on int.
extension IntExtensions on int {
  bool get isEven => this % 2 == 0;

  bool get isOdd => this % 2 == 1;
}

// A widget class (common in Flutter).
class MyWidget extends StatelessWidget {
  final String title;

  const MyWidget({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Text(title);
  }
}

// Stateful widget.
class CounterWidget extends StatefulWidget {
  const CounterWidget({super.key});

  @override
  State<CounterWidget> createState() => _CounterWidgetState();
}

class _CounterWidgetState extends State<CounterWidget> {
  int _count = 0;

  void _increment() {
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text('Count: $_count'),
        ElevatedButton(
          onPressed: _increment,
          child: const Text('Add'),
        ),
      ],
    );
  }
}

// Abstract class.
abstract class Repository<T> {
  Future<T?> getById(int id);
  Future<List<T>> getAll();
  Future<void> save(T item);
}

// Concrete implementation.
class UserRepository implements Repository<User> {
  @override
  Future<User?> getById(int id) async => null;

  @override
  Future<List<User>> getAll() async => [];

  @override
  Future<void> save(User item) async {}
}

// Named constructor.
class Point {
  final double x;
  final double y;

  Point.origin()
      : x = 0,
        y = 0;

  Point(this.x, this.y);

  double distanceTo(Point other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return sqrt(dx * dx + dy * dy);
  }

  static const Point zero = Point(0, 0);
}
